import os
import tweepy
import config
import pandas as pd

API_KEY = config.API_KEY
API_KEY_SECRET = config.SECRET_API_KEY
BEARER_TOKEN = config.MY_BEARER_TOKEN
ACCESS_TOKEN = config.ACCESS_TOKEN
ACCESS_TOKEN_SECRET = config.SECRET_ACCESS_TOKEN

name = 'nytimes'

auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
  
# calling the api 
api = tweepy.API(auth)
  
# fetching the user
user = api.get_user(screen_name=name)
  
# fetching the ID
ID = user.id_str
  
print("The ID of the user is : " + ID)

'''keyword = 'covid'
limit=100

tweets = tweepy.Cursor(api.user_timeline, screen_name=user, q=keyword, tweet_mode='extended').items(limit)

columns = ['Time', 'User', 'Tweet']
data = []

# Iterate through the results and append them to the list
for tweet in tweets:
    data.append([tweet.created_at, tweet.user.screen_name, tweet.full_text])

# Create a dataframe with the results
df = pd.DataFrame(data, columns=columns)

print(df)
'''