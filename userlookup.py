import os
import tweepy
import config
import pandas as pd

API_KEY = config.API_KEY
API_KEY_SECRET = config.SECRET_API_KEY
BEARER_TOKEN = config.MY_BEARER_TOKEN
ACCESS_TOKEN = config.ACCESS_TOKEN
ACCESS_TOKEN_SECRET = config.SECRET_ACCESS_TOKEN

name = 'EPPGroup'

auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
  
# calling the api 
api = tweepy.API(auth)
  
# fetching the user
user = api.get_user(screen_name=name)
  
# fetching the ID
ID = user.id_str
  
print("The ID of the user is : " + ID)