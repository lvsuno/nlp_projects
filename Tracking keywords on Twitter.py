import json
# import pickle
import tweepy
# import csv
from textblob import TextBlob
import preprocess_lvsuno as pl
import cred
import pandas as pd
from pathlib import Path


keywords = ['China', '#python', 'Africa']

# Twitter API credentials
auth = tweepy.OAuthHandler(cred.consumer_key, cred.consumer_secret)
auth.set_access_token(cred.access_token, cred.access_token_secret)
api = tweepy.API(auth)


# Construct a StreamListener class

# API v2
# class Listener(tweepy.StreamingClient):
# API v1
class Listener(tweepy.Stream):
    tweets = []
    limit = 1

    def on_status(self, status):
        print(status.text)
        self.tweets.append(status)
        # print(status.user.screen_name + ": " + status.text)
        if len(self.tweets) == self.limit:
            self.disconnect()

    '''
    def on_errors(self, status_code):
        if status_code == 420:
            print('Error 420')
            # returning False in on_error disconnects the stream
            return False
    '''
    ''' Use this function to perform some data analysis once the data is receive
    def on_data(self, data):
        raw = json.loads(data)
        try:

        except:
            pass
    '''


# API v1
stream_tweet = Listener(cred.consumer_key, cred.consumer_secret, cred.access_token,
                        cred.access_token_secret)

# API v2
# listener = Listener('BEARER_TOKEN')


stream_tweet.filter(track=keywords)

columns = ['Creation_date', 'User', 'Tweet', 'number_of_likes', 'number_of_retweets']
data = []

for tweet in stream_tweet.tweets:
    if not tweet.truncated:
        data.append([tweet.created_at, tweet.user.screen_name, tweet.text, tweet.favorite_count, tweet.retweet_count])
    else:
        data.append([tweet.created_at, tweet.user.screen_name, tweet.extended_tweet['full_text'], tweet.favorite_count,
                     tweet.retweet_count])

df = pd.DataFrame(data, columns=columns)

filename = 'data/Twitter.csv'

if Path(filename).is_file():
    df1 = pd.read_csv(filename, sep=',')
    df = pd.concat([df, df1], ignore_index=True)
else:
    df.to_csv(filename, sep=',')

# Remove duplicates if there are any
df = df.drop_duplicates("text", keep='first')

'''
public_tweets = api.home_timeline()

for tweet in public_tweets:
    print(tweet.text)
'''
