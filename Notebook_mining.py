from Notebook_helperfunctions import *
from tweepy import TweepError

#############################################################################
###                             INPUT                                     ###
#############################################################################
screen_names = ['AP', 'FoxNews', 'nytimes', 'NBCNews', 'CNN']

# command line arguments
args = mining_cml()
small_batch = args.batch
tweet_lim = args.tweet_lim
reset = args.reset
    

# parameters
client = get_twitter_client()
start = datetime.datetime(2016, 10, 1)  
end = datetime.datetime(2017, 1, 31)    
fname_tweet_ids = 'all_ids.json'
f_authorship = 'users/authorship.csv'
total_tweets = []

### GET TWEETS
if not reset:
    print('Getting Tweets...')
    if small_batch:
        for screen_name in screen_names:
            num_tweets = get_user_tweets(client, screen_name, 
                                         tweet_lim=tweet_lim)
            total_tweets.append(num_tweets)
    else: 
        for screen_name in screen_names:
            num_tweets = get_all_user_tweets(screen_name, start, end,
                                         tweet_lim=tweet_lim)
            total_tweets.append(num_tweets)
    print('Found {} tweets.'.format(sum(total_tweets)))


##############################################################################
###                             WRITE RESULTS                              ###
##############################################################################
print('Writing results...')
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
                    writer.writerow([tweet['text'], 
                                     tweet['id'], 
                                     tweet['user']['id']])
        else:
            fin = 'users/{0}/usr_tweetids_{0}.jsonl'.format(screen_name)
            with open(fin, 'r') as f:
                for line in f:
                    ids = json.loads(line)
                    
                    for tweetId in ids:
                        try:
                            tweet = client.get_status(tweetId)
                            writer.writerow([tweet.text, 
                                             tweet.id, 
                                             tweet.user.id])
                        except TweepError as e:
                            print(e)
                            time.sleep(60*15)


print('done writing results.\nCheck: {}'.format(f_authorship))


##############################################################################
###                                 FUTURE USE                             ###
##############################################################################
# Create pkl_objects to store classifiers
dest = os.path.join('pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
    print('creating: {}'.format(dest))


# Create figures to sore results
dest = os.path.join('figures')
if not os.path.exists(dest):
    os.makedirs(dest)
