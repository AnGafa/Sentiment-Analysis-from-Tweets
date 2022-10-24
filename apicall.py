import tweepy
import pandas as pd
import config
from sqlalchemy import create_engine


def get_tweets():
    
    client = tweepy.Client(bearer_token=config.MY_BEARER_TOKEN)
    search_query = config.search_query

    start_time = "2021-12-10T00:00:00Z"

    tweets = client.search_recent_tweets(query=search_query,
                                        username = "nytimes",
                                        start_time=start_time,
                                        max_results = config.max_results,
                                        expansions="author_id",
                                        tweet_fields = ["author_id", "created_at", "text"],
                                        user_fields = ["id", "location"])

    tweet_info_ls = []

    for tweet, user in zip(tweets.data, tweets.includes['users']):
        tweet_info = {
            'created_at': tweet.created_at,
            'text': tweet.text,
            'location': user.location,
        }
        tweet_info_ls.append(tweet_info)

    tweets_df = pd.DataFrame(tweet_info_ls)
    return tweets_df

try:
    tweetData = get_tweets()
    #engine = create_engine('mysql+mysqldb://root:test123@localhost:1234/twitterScrape', echo=False)
    #tweetData.to_sql(name='tweets', con=engine, if_exists='append', index=False)
except Exception as e:
    print("Error while connecting to MySQL", e)