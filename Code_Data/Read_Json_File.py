import json
import pandas as pd
import matplotlib.pyplot as plt
import datetime

tweets_data_path = 'tweet_data_testing.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")

# reading line by line from json file
for line in tweets_file:
    try:
        tweet = json.loads(line)

        # changing Twitter time format to python yy,m,d

        tweet_daytime = datetime.datetime.fromtimestamp(int(tweet['timestamp_ms']) / 1000)

        tweet_day = tweet_daytime.strftime('%Y-%m-%d')

        # print(tweet_day)
        # appendign to the tweets_data array
        tweets_data.append(tweet)
    except:
        continue

    tweets = pd.DataFrame(tweets_data)




tweets = pd.DataFrame(tweets_data)
print(tweets.info())

#tweets.to_csv('tweets.cvs')

df = pd.read_csv('tweets.cvs')

chosen_Df = pd.DataFrame(columns=['Date','Tweet','favorite_count'])

print (chosen_Df.info())
# sending data to the new dataframe

chosen_Df[['Date','Tweet','favorite_count']] = df[['created_at','text','favorite_count']]

# here Date is replacing with formated date
chosen_Df['Date'] = tweet_day

chosen_Df.to_csv('ReadyTweet.cvs')
print(chosen_Df)
print(chosen_Df['Tweet'])
print (chosen_Df.info())






#print (tweets['created_at'].values)

#print (tweets['text'].values)



