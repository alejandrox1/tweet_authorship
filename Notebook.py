import pyprind
import os
import sys
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix

from config import *

dest = os.path.join('pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
    print(dest)

### TWITTER AUTHENTICATION
def get_twitter_auth():
    """Setup Twitter Authentication.
    
    Return: tweepy.OAuthHandler object
    """
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth
    
def get_twitter_client():
    """Setup Twitter API Client.
    
    Return: tweepy.API object
    """
    auth = get_twitter_auth()
    client = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
    return client

### FILE MANAGEMENT
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
    
    Return: string
    """
    url1 = 'https://twitter.com/search?f=tweets&q=from%3A'
    url2 = screen_name + '%20since%3A' + start.strftime('%Y-%m-%d') 
    url3 = ''
    if no_rt:
        url3 = '%20until%3A' + end.strftime('%Y-%m-%d') + '%20&src=typd'
    else:
        url3 = '%20until%3A' + end.strftime('%Y-%m-%d') + '%20include%3Aretweets&src=typd'
    
    return url1 + url2 + url3
    
def increment_day(date, i):
    """Increment day object by i days.
    
    Params
    -------
    date : datetime-obj
    i : int
    
    Return: datetime object
    """
    return date + datetime.timedelta(days=i)

### GETTTING TWEETS
def get_user_tweets(screen_name, no_rt=True):
    """Get tweets for a given user (3,200 limit)
    
    Create a subdir named 'users'.
    In this subdir, a jsonl file will store all the tweets writen
    by the given user.
    
    Params
    -------
    screen_name : str    
    """
    # Make dir structure
    makedir(screen_name)

    total_tweets = 0
    fname = 'users/{0}/usr_timeline_{0}.jsonl'.format(screen_name)
    with open(fname, 'a') as f:
        for page in Cursor(client.user_timeline, screen_name=screen_name, count=200).pages(5): #16): 
            for tweet in page:
                total_tweets += 1
                if no_rt:
                    if not tweet.retweeted and 'RT @' not in tweet.text:
                        f.write(json.dumps(tweet._json)+'\n')
                else:
                    f.write(json.dumps(tweet._json)+'\n')
    return total_tweets

def get_all_user_tweets(screen_name, start, end, no_rt=True):
    """
    Params
    ------
    screen_name : str
    start : datetime-obj
    end : datetime-obj
    no_rt : bool
    
    """
    # Special parameters
    fname_tweet_ids = 'users/{0}/usr_tweetids_{0}.jsonl'.format(screen_name)
    
    # Make dir structure
    makedir(screen_name)
    
    # Selenium parames
    delay = 1  # time to wait on each page load before reading the page
    driver = webdriver.Chrome() 
    tweet_selector = 'li.js-stream-item'
    id_selector = '.time a.tweet-timestamp'
    
    ids_total = 0
    for day in range((end - start).days + 1):
        # Get Twitter search url
        startDate = increment_day(start, 0)
        endDate = increment_day(start, 1)
        url = twitter_url(screen_name, no_rt, startDate, endDate)

        driver.get(url)
        time.sleep(delay)
        
        try:
            found_tweets = driver.find_elements_by_css_selector(tweet_selector)
            increment = 10

            # Scroll through the Twitter search page
            while len(found_tweets) >= increment:
                print('scrolling down to load more tweets')
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                time.sleep(delay)
                found_tweets = driver.find_elements_by_css_selector(tweet_selector)
                increment += 10

            # Get the IDs for all Tweets
            ids = []
            with open(fname_tweet_ids, 'a') as fout:
                for tweet in found_tweets:
                    try:
                        tweet_id = tweet.find_element_by_css_selector(
                                    id_selector).get_attribute('href').split('/')[-1]
                        ids.append(tweet_id)
                        ids_total += 1
                    except StaleElementReferenceException as e:
                        print('lost element reference', tweet)
                        
                # Save ids to file
                data_to_write = list(set(ids))
                fout.write(json.dumps(data_to_write)+'\n')
            print('{} tweets found, {} total'.format(len(found_tweets), ids_total))
        
        except NoSuchElementException:
            print('no tweets on this day')

        start = increment_day(start, 1)
    
    # Close selenium driver
    driver.close()
    
    return ids_total


########### INPUT
client = get_twitter_client()
print('Started')
screen_names = ['AP', 'FoxNews', 'nytimes']
small_batch = True
start = datetime.datetime(2017, 1, 10)  
end = datetime.datetime(2017, 1, 16)    
fname_tweet_ids = 'all_ids.json'
total_tweets = []

########### GET TWEETS
print('Downloading Tweets: Chekc users/')
if small_batch:
    for screen_name in screen_names:
        num_tweets = get_user_tweets(screen_name)
        total_tweets.append(num_tweets)
else: 
    for screen_name in screen_names:
        num_tweets = get_all_user_tweets(screen_name, start, end, no_rt=True)
        total_tweets.append(num_tweets)

########### CLEAN UP RESULTS
f_authorship = 'users/authorship.csv'

print('Cleaning downloaded tweets')
#pbar = pyprind.ProgBar(sum(total_tweets))
with open(f_authorship, 'w') as fout:
    writer = csv.writer(fout)
    # Header
    writer.writerow(['text','id','user_id'])

    for screen_name in screen_names:
        if small_batch:
            fin = 'users/{0}/usr_timeline_{0}.jsonl'.format(screen_name)
            with open(fin, 'r') as f:
                for line in f:
                    tweet = json.loads(line)
                    writer.writerow([tweet['text'], tweet['id'], tweet['user']['id']])
                    #fout.write('"{0}",{1},{2}\n'.format(tweet['text'].encode("utf-8"), tweet['id'], tweet['user']['id']))
                    #pbar.update()
        else:
            fin = 'users/{0}/usr_tweetids_{0}.jsonl'.format(screen_name)
            with open(fin, 'r') as f:
                for line in f:
                    ids = json.loads(line)
                    
                    for tweetId in ids:
                        tweet = client.get_status(tweetId)
                        writer.writerow([tweet.text, tweet.id, tweet.user.id])
                        #fout.write('"{0}",{1},{2}\n'.format(tweet.text.encode("utf-8"), tweet.id, tweet.user.id))
                        #pbar.update()

########### PREPROCESSING
df = pd.read_csv(f_authorship)
df.drop_duplicates()

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

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
#df['text'] = df['text'].apply(preprocessor)

X = df.loc[:, 'text'].values
y = df.loc[:, 'user_id'].values
le = LabelEncoder()
y = le.fit_transform(y)
print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


########### BUILDING MODELS
########### LOGIT
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

param_grid = [{'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

gs_lr_tfidf = GridSearchCV(lr_tfidf, 
                           param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

lg_time0 = time.time()
gs_lr_tfidf.fit(X_train, y_train)
lg_time1 = time.time()
print('EXECUTION TIME for logit gs : {} secs\n\n'.format(lg_time1 - lg_time0))
pickle.dump(gs_lr_tfidf, open(os.path.join(dest, 'gs_logit.pkl'), 'wb'), protocol=4)

print('Best parameter set: {} \n'.format(gs_lr_tfidf.best_params_))
print('CV Accuracy: {:.3f}'.format(gs_lr_tfidf.best_score_))
clf_lr = gs_lr_tfidf.best_estimator_
print('Test Accuracy: {:.3f}'.format(clf_lr.score(X_test, y_test)))


########### SVC
from sklearn.svm import SVC

svc_tfidf = Pipeline([('vect', tfidf),
                     ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'clf__C': param_range, 
               'clf__kernel': ['linear']},
              {'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__C': param_range, 
               'clf__kernel': ['linear']},
              {'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'clf__C': param_range, 
               'clf__gamma': param_range, 
               'clf__kernel': ['rbf']},
              {'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__C': param_range, 
               'clf__gamma': param_range, 
               'clf__kernel': ['rbf']}]

gs_svc_tfidf = GridSearchCV(estimator=svc_tfidf, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=5,
                  verbose=1,
                  n_jobs=-1)

svc_time0 = time.time()
gs_svc_tfidf.fit(X_train, y_train)
svc_time1 = time.time()
print('EXECUTION TIME for svc gs : {} secs\n\n'.format(svc_time1 - svc_time0))
pickle.dump(gs_svc_tfidf, open(os.path.join(dest, 'gs_svc.pkl'), 'wb'), protocol=4)

print('Best parameter set: {} \n'.format(gs_svc_tfidf.best_params_))
print('CV Accuracy: {:.3f}'.format(gs_svc_tfidf.best_score_))
clf_svc = gs_svc_tfidf.best_estimator_
print('Test Accuracy: {:.3f}'.format(clf_svc.score(X_test, y_test)))


########### NAIVE BAYES
from sklearn.naive_bayes import MultinomialNB

nb_tfidf = Pipeline([('vect', tfidf),
                    ('clf', MultinomialNB())])

param_range = [0.25, 0.5, 0.75, 1.0]
param_grid = [{'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'clf__alpha': param_range},
              {'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__alpha': param_range}]
                   
gs_nb_tfidf = GridSearchCV(estimator=svc_tfidf, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=5,
                  verbose=1,
                  n_jobs=-1)

nb_time0 = time.time()
gs_nb_tfidf.fit(X_train, y_train)
nb_time1 = time.time()
print('EXECUTION TIME for nb gs : {} secs\n\n'.format(nb_time1 - nb_time0))
pickle.dump(gs_nb_tfidf, open(os.path.join(dest, 'gs_nb.pkl'), 'wb'), protocol=4)

print('Best parameter set: {} \n'.format(gs_nb_tfidf.best_params_))
print('CV Accuracy: {:.3f}'.format(gs_nb_tfidf.best_score_))
clf_nb = gs_nb_tfidf.best_estimator_
print('Test Accuracy: {:.3f}'.format(clf_nb.score(X_test, y_test)))


########### SGD
from sklearn.linear_model import SGDClassifier

sgd_tfidf = Pipeline([('vect', tfidf),
                     ('clf', SGDClassifier(random_state=42)),])

param_range = [0.25, 0.5, 0.75, 1.0]
param_grid = [{'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'clf__loss' : ['hinge', 'log'],
               'clf__penalty' : ['l1', 'l2'],
               'clf__n_iter' : [3,5,7],
               'clf__alpha': param_range},
              {'vect__ngram_range': [(1, 2)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter, tokenizer_twitter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__loss' : ['hinge', 'log'],
               'clf__penalty' : ['l1', 'l2'],
               'clf__n_iter' : [3,5,7],
               'clf__alpha': param_range}]

gs_sgd_tfidf = GridSearchCV(estimator=sgd_tfidf, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=5,
                  verbose=1,
                  n_jobs=-1)

sgd_time0 = time.time()
gs_sgd_tfidf.fit(X_train, y_train)
sgd_time1 = time.time()
print('EXECUTION TIME for sgd gs : {} secs\n\n'.format(sgd_time1 - sgd_time0))
pickle.dump(gs_sgd_tfidf, open(os.path.join(dest, 'gs_sgd.pkl'), 'wb'), protocol=4)

print('Best parameter set: {} \n'.format(gs_sgd_tfidf.best_params_))
print('CV Accuracy: {:.3f}'.format(gs_sgd_tfidf.best_score_))
clf_sgd = gs_sgd_tfidf.best_estimator_
print('Test Accuracy: {:.3f}'.format(clf_sgd.score(X_test, y_test)))


