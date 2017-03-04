import os
import sys
import argparse
import time
import datetime
import re
import json
import csv
import pickle

import numpy as np
import pandas as pd
import dask.dataframe as dd

from tweepy import API
from tweepy import OAuthHandler
from tweepy import Cursor
from tweepy import TweepError

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException 
from selenium.common.exceptions import StaleElementReferenceException

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from config import *

# COMMAND LINE OPTIONS
def gs_cml():
    """Command line arguments for gridsearches

    Params
    ------
    infile : str {default users/authorship.csv}
            Contains tweets, tweet_id, user_id.
    ngram : int {default 1}
            n-gram. Default is `1`, an unigram .
    jobs : int {default -1}
            Number of CPUs to run gridsearches on.
            Default is `-1` which uses all detected CPUs.

    Returns
    -------
    {argparse-obj} cml arguments container.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="input file",
                        default="users/authorship.csv", type=str)
    parser.add_argument("--ngram",  help="max size for n-grams", 
                        default=1,                      type=int)
    parser.add_argument("--jobs",   help="number of CPUs", 
                        default=-1,                     type=int)
    args = parser.parse_args()
    return args

def mining_cml():
    """Command line arguments for mining.                                  
                                                                                
    Params                                                                      
    ------                                                                      
    batch : bool {default True}                                                     
            if True: use twitter's API, else use selenium.                                
    tweets_lim : int {default -1}
            Maximum number of tweets to be mined. 
            Default `-1` will either be limited by the twitter API or by
            the range of dates provided.
                                                                                
    Returns                                                                     
    -------                                                                     
    {argparse-obj} cml arguments container.                                     
    """
    parser = argparse.ArgumentParser()                                          
    parser.add_argument("--batch",      help="Mine a small batch of tweets", 
                        default='y', type=str)                                               
    parser.add_argument("--tweet_lim",  help="Max number of tweets to mine",
                        default=-1,  type=int)                                               
    args = parser.parse_args()                                                  
    return args 


# TWITTER AUTHENTICATION
def get_twitter_auth():
    """Setup Twitter Authentication.
    
    Returns
    --------
    {tweepy.OAuthHandler}
    """
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth


def get_twitter_client():
    """Setup Twitter API Client.
    
    Return
    -------
    {tweepy.API object}
    """
    auth = get_twitter_auth()
    client = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True,
                 compression=True)
    return client


# FILE MANAGEMENT
def makedir(screen_name):
    """Create subdirectory 'users/screen_name' to store mined data.
    
    Params
    -------
    screen_name : str
    """
    dirname = 'users/{}'.format(screen_name)

    try:
        os.makedirs(dirname, mode=0o755, exist_ok=True)
    except OSError:
        print('Directory {} already exists.'.format(dirname))
    except Exception as e:
        print('Error while creating directory {}'.format(dirname))
        print(e)
        sys.exit(1)


def twitter_url(screen_name, no_rt, start, end):
    """Form url to access tweets via Twitter's search page.

    Params
    -------
    screen_name : str
    no_rt : bool
    start : datetime-onj
    end : datetime-obj
    
    Returns
    -------
    {string} search url for twitter
    """
    url1 = 'https://twitter.com/search?f=tweets&q=from%3A'
    url2 = screen_name + '%20since%3A' + start.strftime('%Y-%m-%d') 
    url3 = ''
    if no_rt:
        url3 = '%20until%3A' + end.strftime('%Y-%m-%d') + '%20&src=typd'
    else:
        url3 = '%20until%3A' + end.strftime('%Y-%m-%d') + \
                '%20include%3Aretweets&src=typd'
    
    return url1 + url2 + url3


def increment_day(date, i):
    """Increment day object by i days.
    
    Params
    -------
    date : datetime-obj
    i : int
    
    Returns
    -------
    {datetime-obj} next day.
    """
    return date + datetime.timedelta(days=i)


# GETTTING TWEETS
def get_user_tweets(client, screen_name, tweet_lim=3200, no_rt=True):
    """Get tweets for a given user (3,200 limit)
    
    Create a subdir named 'users'.
    In this subdir, a jsonl file will store all the tweets writen
    by the given user.
    
    Params
    -------
    client : tweepy.api
    screen_name : str  
    no_rt : bool {default True}
    tweet_lim : int {default 3,200}

    returns
    -------
    {int} number of tweets mined
    """
    # Make dir structure
    makedir(screen_name)

    total_tweets = 0
    fname = 'users/{0}/usr_timeline_{0}.jsonl'.format(screen_name)
    with open(fname, 'a') as f:
        for page in Cursor(client.user_timeline, screen_name=screen_name,
                           count=200).pages(16): 
            for tweet in page:
                total_tweets += 1
                if no_rt:
                    if not tweet.retweeted and 'RT @' not in tweet.text:
                        f.write(json.dumps(tweet._json)+'\n')
                else:
                    f.write(json.dumps(tweet._json)+'\n')

                # break if tweet_lim has been reached
                if total_tweets == tweet_lim:
                    return total_tweets
    return total_tweets


def get_all_user_tweets(screen_name, start, end, tweet_lim=3200, no_rt=True):
    """
    Params
    ------
    screen_name : str
    start : datetime-obj
    end : datetime-obj
    no_rt : bool
    tweet_lim : int {default 3,200}
    
    returns
    -------
    {int} total number of tweet ids obtained
    """
    # Make dir structure                                                        
    makedir(screen_name)

    # name of file for saving tweet ids
    fname_tweet_ids = 'users/{0}/usr_tweetids_{0}.jsonl'.format(screen_name)
    
    # Selenium parames
    delay = 1  # time to wait on each page load before reading the page
    driver = webdriver.Chrome() 
    
    ids_total = 0
    for day in range((end - start).days + 1):
        # Get Twitter search url
        startDate = increment_day(start, 0)
        endDate = increment_day(start, 1)
        url = twitter_url(screen_name, no_rt, startDate, endDate)

        driver.get(url)
        time.sleep(delay)
        
        try:
            found_tweets = \
            driver.find_elements_by_css_selector('li.js-stream-item')
            increment = 10

            # Scroll through the Twitter search page
            while len(found_tweets) >= increment:
                # scroll down for more results
                driver.execute_script(
                    'window.scrollTo(0, document.body.scrollHeight);'
                )
                time.sleep(delay)
                # select tweets
                found_tweets = driver.find_elements_by_css_selector(
                    'li.js-stream-item'
                )
                increment += 10

            # Get the IDs for all Tweets
            ids = []
            with open(fname_tweet_ids, 'a') as fout:
                for tweet in found_tweets:
                    try:
                        # get tweet id
                        tweet_id = tweet.find_element_by_css_selector(
                            '.time a.tweet-timestamp'
                        ).get_attribute('href').split('/')[-1]
                        ids.append(tweet_id)
                        ids_total += 1

                        # break if tweet_lim has been reached                           
                        if ids_total == tweet_lim:                                   
                            return ids_total

                    except StaleElementReferenceException as e:
                        print('lost element reference', tweet)
                        
                # Save ids to file
                data_to_write = list(set(ids))
                fout.write(json.dumps(data_to_write)+'\n')
        
        except NoSuchElementException:
            print('no tweets on this day')

        start = increment_day(start, 1)
    
    # Close selenium driver
    driver.close()
    print('{} tweets found total'.format(ids_total))
    return ids_total


### PREPROCESSING
stop = stopwords.words('english')
def preprocessor(text):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text) + ' '.join(emoticons).replace('-', '')
    return text


def tokenizer(text):
    return text.split()


porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


tweet_token = TweetTokenizer()
def tokenizer_twitter(text):
    return tweet_token.tokenize(text)

