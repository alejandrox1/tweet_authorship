from Notebook_helperfunctions import *


### INPUT
client = get_twitter_client()
    
screen_names = ['AP', 'FoxNews', 'nytimes']
small_batch = True
    
start = datetime.datetime(2017, 1, 10)  
end = datetime.datetime(2017, 1, 16)    
fname_tweet_ids = 'all_ids.json'
total_tweets = []
f_authorship = 'users/authorship.csv'
    
### GET TWEETS
print('Getting Tweets...')
if small_batch:
    for screen_name in screen_names:
        num_tweets = get_user_tweets(client, screen_name)
        total_tweets.append(num_tweets)
else: 
    for screen_name in screen_names:
        num_tweets = get_all_user_tweets(screen_name, start, end, no_rt=True)
        total_tweets.append(num_tweets)
print('Found {} tweets.'.format(sum(total_tweets)))

### WRITE RESULTS
#pbar = pyprind.ProgBar(sum(total_tweets))
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
                    writer.writerow([tweet['text'], tweet['id'], tweet['user']['id']])
                    #fout.write('"{0}",{1},{2}\n'.format(
                    #tweet['text'].encode("utf-8"), tweet['id'], tweet['user']['id']))
                    #pbar.update()
        else:
            fin = 'users/{0}/usr_tweetids_{0}.jsonl'.format(screen_name)
            with open(fin, 'r') as f:
                for line in f:
                    ids = json.loads(line)
                    
                    for tweetId in ids:
                        tweet = client.get_status(tweetId)
                        writer.writerow([tweet.text, tweet.id, tweet.user.id])
                        #fout.write('"{0}",{1},{2}\n'.format(
                        #tweet.text.encode("utf-8"), tweet.id, tweet.user.id))
                        #pbar.update()
print('done writing results.\nCheck: {}'.format(f_authorship))

# Create pkl_objects to store classifiers
dest = os.path.join('pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
    print('creating: {}'.format(dest))

