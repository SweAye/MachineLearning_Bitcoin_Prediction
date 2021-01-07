import pandas as pd
import numpy as np
from tweepy.streaming import StreamListener # for streamming data from the twitter
from tweepy import OAuthHandler # to authenticate the access
from tweepy import Stream
import Twitter_credential # that where I created the authentication secet key


# Twitter Client



#This handle twitter authentication and connection to the twitter

class TwitterStreammer():
    """
    streamming and processing live tweets

    """
    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweet_filename, hash_tag_list ):
        listener = StdOutListener(fetched_tweet_filename)
        auth = OAuthHandler(Twitter_credential.CONSUMER_KEY, Twitter_credential.CONSUMER_SECRET)
        auth.set_access_token(Twitter_credential.ACCESS_TOKEN, Twitter_credential.ACCESS_TOKEN_SECRET)

        stream = Stream(auth, listener)

        stream.filter(track= hash_tag_list)




class StdOutListener (StreamListener): # StreamListener obj is passed from the tweep.streamming module
    """
    listener class to print out the streamed live tweet
    """

    def __init__(self, fetched_tweets_filename):# to save the streammed tweets
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        with open(self.fetched_tweets_filename,'a') as tf:
            tf.write(data)
        return True




    def on_error(self, status):
        print(status)

if __name__== "__main__":

    hash_tag_list = ["bitcoin"]# create the list for twitter hash tag

    fetched_tweets_filename = "tweets.json"
    #twitter_Streamer = TwitterStreammer()
   # twitter_Streamer.stream_tweets( fetched_tweets_filename,hash_tag_list)



