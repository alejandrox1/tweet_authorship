from Notebook_helperfunctions import *
from tweepy import TweepError

#############################################################################
###                             INPUT                                     ###
#############################################################################
screen_names = ['X1alejandro3x', 'HEPfeickert']

# command line arguments
args = mining_cml()
verbosity = args.verbose
virtuald = args.virtual
f_authorship = args.outfile
small_batch = args.smallbatch
tweet_lim = args.tweet_lim
reset = args.reset

# parameters
client = get_twitter_client()
start = datetime.datetime(2013, 1, 1)  
end = datetime.datetime.today()    
fname_tweet_ids = 'all_ids.json'
total_tweets = []

### GET TWEETS
if not reset:
    print('Getting Tweets...')
    if small_batch: # use tweepy
        for screen_name in screen_names:
            num_tweets = get_user_tweets(client, 
                                         screen_name, 
                                         tweet_lim=tweet_lim)
            total_tweets.append(num_tweets)

    else: # use selenium extension
        for screen_name in screen_names:
            num_tweets = get_all_user_tweets(screen_name, 
                                             start, end,
                                             tweet_lim=tweet_lim, 
                                             virtuald=virtuald)
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
        if small_batch: # use tweepy
            fin = 'users/{0}/usr_timeline_{0}.jsonl'.format(screen_name)
            with open(fin, 'r') as f:
                for line in f:
                    tweet = json.loads(line)
                    writer.writerow([tweet['text'], 
                                     tweet['id'], 
                                     tweet['user']['id']])

        else: # use selenium extension
            fin = 'users/{0}/usr_tweetids_{0}.jsonl'.format(screen_name)
            fcheck = 'users/{0}/checkpoints_{0}.txt'.format(screen_name)
            if not os.path.isfile(fcheck): # if no checkpoint file
                with open(fin, 'r') as f, open(fcheck, 'w') as c:
                    for line in iter(f.readline, ''):
                        # save the location of file
                        c.write( '{}\n'.format(f.tell()) )
                        # load ids
                        ids = json.loads(line)
                    
                        for tweetId in ids:
                            try:
                                tweet = client.get_status(tweetId)
                                writer.writerow([tweet.text, 
                                                tweet.id, 
                                                tweet.user.id])
                            except TweepError as e:
                                if verbosity:
                                    print(e)
                                time.sleep(60*15)

            else: # if checkpoints file allready exists
                with open(fin, 'r') as f, open(fcheck, 'r+') as c:    
                    checkpoints = c.readlines()
                    checkpoints = [check.strip('\n') for check in checkpoints 
                                   if check.strip('\n')!='']
                    # go to last checkpoint
                    if checkpoints:
                        f.seek(int(checkpoints[-1]))
                    for line in iter(f.readline, ''):                                              
                        # save the location of file
                        c.write( '{}\n'.format(f.tell()) )
                        # load ids
                        ids = json.loads(line)                                  
                                                                                
                        for tweetId in ids:                                     
                            try:                                                
                                tweet = client.get_status(tweetId)              
                                writer.writerow([tweet.text,                    
                                                tweet.id,                       
                                                tweet.user.id])                 
                            except TweepError as e:                             
                                if verbosity:
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
