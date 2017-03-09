import os
import sys
import argparse
import time
import datetime
import re
import string
import json
import csv
import pickle

from tweepy import API
from tweepy import OAuthHandler
from tweepy import Cursor

from xvfbwrapper import Xvfb
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException 
from selenium.common.exceptions import StaleElementReferenceException

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer

from config import *

# COMMAND LINE OPTIONS
def gs_cml():
    """Command line arguments for gridsearches

    Options
    -------
    infile : str (default users/authorship.csv)
            Contains tweets, tweet_id, user_id.
    ngram : int (default 1)
            n-gram. Default is `1`, an unigram .
    jobs : int (default -1)
            Number of CPUs to run gridsearches on.
            Default is `-1` which uses all detected CPUs.

    Returns
    -------
    {argparse-obj} cml arguments container.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", 
                        help="input file (defaut: users/authorship.csv)",
                        default="users/authorship.csv", 
                        type=str)
    parser.add_argument("--ngram",  
                        help="max size for n-grams (default: 1)", 
                        default=1,                      
                        type=int)
    parser.add_argument("--jobs",   
                        help="number of CPU cores to use (default: -1).", 
                        default=-1,                     
                        type=int)
    args = parser.parse_args()
    return args


def mining_cml():
    """Command line arguments for mining.                                  
                                                                                
    Options                                                                      
    --------                 
    outfile : str (default users/authorship.csv)                                 
                Contains tweets, tweet_id, user_id.
    batch : str (default 'y')                                                     
            If 'y': use twitter's API, else use selenium.                                
    tweets_lim : int (default -1)
            Maximum number of tweets to be mined. 
            Default `-1` will either be limited by the twitter API or by
            the range of dates provided.
    reset : str (default 'n')
            If 'y': skip tweet acquisition and go straight to data cleaning.
            If 'y': batch is automatically set to False.
                                                                                
    Returns                                                                     
    -------                                                                     
    {argparse-obj} cml arguments container.                                     
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", 
                        help="increase output verbosity",        
                        action="store_true")
    parser.add_argument("-d", "--virtual",                                      
                        help="Use virtual display (use on EC2 instance).",                       
                        action="store_true")
    parser.add_argument("-o", "--outfile", 
                        help="output file (defaut: users/authorship.csv)",       
                        default="users/authorship.csv",                         
                        type=str)
    parser.add_argument("-s", "--smallbatch",      
                        help="Mine small batch of tweets using twitter's API.", 
                        action="store_true")                                               
    parser.add_argument("-l", "--tweet_lim",  
                        help="Max number of tweets to mine (default: -1).",
                        default=-1,  
                        type=int)
    parser.add_argument("-r", "--reset",  
                        help="Reset connection to continue getting tweets ",    
                        action="store_true")
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
    client = API(auth, 
                 wait_on_rate_limit=True, 
                 wait_on_rate_limit_notify=True,
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


def get_all_user_tweets(screen_name, start, end, tweet_lim=3200, no_rt=True,
                       virtuald=False):
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
   
    # Headless displays with Xvfb (X virtual framebuffer)
    if virtuald:
        vdisplay = Xvfb()
        vdisplay.start()
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
                            print('{} tweets found.'.format(ids_total))
                            return ids_total

                    except StaleElementReferenceException as e:
                        print('Lost element reference.', tweet)
                        
                # Save ids to file
                data_to_write = list(set(ids))
                fout.write(json.dumps(data_to_write)+'\n')
        
        except NoSuchElementException:
            print('No tweets found.')

        start = increment_day(start, 1)
    
    # Close selenium driver
    driver.close()
    if virtuald:
        vdisplay.stop()
    print('{} tweets found.'.format(ids_total))
    return ids_total


### PREPROCESSING
stop = stopwords.words('english')
def preprocessor(doc):
   """Proportion of characters in document                                     
                                                                                
    :( :) :P :p :O :3 :| :/ :\ :$ :* :@                                         
    :-( :-) :-P :-p :-O :-3 :-| :-/ :-\ :-$ :-* :-@                             
    :^( :^) :^P :^p :^O :^3 :^| :^/ :^\ :^$ :^* :^@                             
    ): (: $: *:                                                                 
    )-: (-: $-: *-:                                                             
    )^: (^: $^: *^:                                                             
    <3 </3 <\3                                                                  
    o.O O.O O.o                                                                 
    :smile: :hug: :pencil:                                                      
    """                                                                         
    re_url = r"(http|https):\/\/.\S+"                                           
    re_emoji = r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?"\
    "[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)"    
    # remove urls                                                               
    doc = re.sub(re_url, "", doc)                                               
    # remove emoticons                                                          
    doc = re.sub(re_emoji, "", doc)                                             
    # remove emojis   
    try:
        # UCS-4
        extra = re.compile(u'[U00010000-U0010ffff]')
        doc = extra.sub('', doc)
    except re.error:
        # UCS-2
        extra = re.compile(u'[uD800-uDBFF][uDC00-uDFFF]')
        doc = extra.sub('', doc)
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    doc = regex.sub('', doc)
    
    return doc


def tokenizer(text):
    return text.split()


porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


tweet_token = TweetTokenizer()
def tokenizer_twitter(text):
    return tweet_token.tokenize(text)

