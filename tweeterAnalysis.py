import tweepy
import pandas as pd

consumer_secret = 'UQiB7ubGC3LTWQCHSI6limjbrIbkSma4i06yRYstVW1liMUugs' #Your API/Consumer Secret Key
consumer_key = '43EOcQFPV47wPGMbl2vKkWbev' #Your API/Consumer key 
access_token = '1238712230989848576-UfrhCiWoAni9r1EuiDawJgVPyYLbMA'    #Your Access token key
access_token_secret = 'hd9lAB5YwMajBTzs5xZL3X9YXP04oP7Xmmx2qTgpgIw0a' #Your Access token Secret key

client_id = 'alNETGFGbjlwSGRQS2FHanc1enY6MTpjaQ'
client_secret = 'W7-LL_f2hvehcJxAdvL4WOlh2KDFT-8GzEKgqi_WGtCv1z3lh1'

#Pass in our twitter API authentication key
auth = tweepy.OAuth2AppHandler(
    consumer_key, consumer_secret
)

api = tweepy.API(auth)


search_query = "'ref''world cup'-filter:retweets AND -filter:replies AND -filter:links"
no_of_tweets = 100

try:
    #The number of tweets we want to retrieved from the search
    tweets = api.search_tweets(q=search_query, lang="en", count=no_of_tweets, tweet_mode ='extended')
    
    #Pulling Some attributes from the tweet
    attributes_container = [[tweet.user.name, tweet.created_at, tweet.favorite_count, tweet.source, tweet.full_text] for tweet in tweets]

    #Creation of column list to rename the columns in the dataframe
    columns = ["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"]
    
    #Creation of Dataframe
    tweets_df = pd.DataFrame(attributes_container, columns=columns)
except BaseException as e:
    print('Status Failed On,',str(e))