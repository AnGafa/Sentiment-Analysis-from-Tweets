import os
import tweepy
import config
import pandas as pd

API_KEY = config.API_KEY
API_KEY_SECRET = config.SECRET_API_KEY
BEARER_TOKEN = config.MY_BEARER_TOKEN
ACCESS_TOKEN = config.ACCESS_TOKEN
ACCESS_TOKEN_SECRET = config.SECRET_ACCESS_TOKEN

client = tweepy.Client( bearer_token= BEARER_TOKEN, 
                        consumer_key=API_KEY, 
                        consumer_secret=API_KEY_SECRET, 
                        access_token=ACCESS_TOKEN, 
                        access_token_secret=ACCESS_TOKEN_SECRET,
                        return_type='json',
                        wait_on_rate_limit=True)
  
query = 'from:xQc'

# get max. 100 tweets
tweets = client.get_list_tweets(query=query, tweet_fields=['author_id'], max_results=10)

