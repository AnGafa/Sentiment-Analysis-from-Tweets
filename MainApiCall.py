import os
from xmlrpc.client import DateTime
from tomlkit import datetime
import tweepy
import config
import pandas as pd
import pyodbc
import datetime as dt
import rfc3339
import iso8601

API_KEY = config.API_KEY
API_KEY_SECRET = config.SECRET_API_KEY
BEARER_TOKEN = config.MY_BEARER_TOKEN
ACCESS_TOKEN = config.ACCESS_TOKEN
ACCESS_TOKEN_SECRET = config.SECRET_ACCESS_TOKEN
MAX_RESULTS = config.MAX_RESULTS
LATEST_DATE = config.LATEST_DATE

client = tweepy.Client( bearer_token= BEARER_TOKEN, 
                        consumer_key=API_KEY, 
                        consumer_secret=API_KEY_SECRET, 
                        access_token=ACCESS_TOKEN, 
                        access_token_secret=ACCESS_TOKEN_SECRET,
                        return_type='json',
                        wait_on_rate_limit=True)

#converts the date to the rfc format
def convertToDatetime(latestDate):
    def get_date_object(date_string):
        return iso8601.parse_date(date_string)
    
    def get_date_string(date_object):
        return rfc3339.rfc3339(date_object)
    
    test_date = get_date_object(latestDate)
    datetime = get_date_string(test_date)
    
    return datetime

#connect to db
def dfConn():
    try:
        conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=Z690-ELDER;'
                          'Database=SentimentAnalysis;'
                          'Trusted_Connection=yes;')
    except pyodbc.Error as e:
        print("Error while connecting to db", e)
    return conn

conn = dfConn()

def dbRead(conn):
    try:
        cursor = conn.cursor()
        
        idList = []
        
        #get all the users from the database
        cursor.execute("SELECT TwitterId FROM dbo.Users")
        for row in cursor:
            idList.append(list(map(int, list(row)))) #convert to int and append to list
        
        cursor.close()
            
    except pyodbc.Error as e:
        print("Error while reading db", e)
        
    return idList

idList = dbRead(conn)

#convert the latest date to a datetime object
#this is the latest date that the tweets will be fetched from to prevent duplicates
datetime = convertToDatetime(LATEST_DATE)

data = []

#call the api and get the tweets for each user
for id in idList:
    tweets = client.get_users_tweets(id=id[0], tweet_fields=['author_id', 'created_at'], max_results=100, start_time=datetime)

    # Iterate through the results and append them to the list
    if tweets[0]:
        for tweet in tweets[0]:
            data.append([tweet['created_at'], tweet['author_id'], tweet['text']])

# Create a dataframe with the results
columns = ['Date', 'User', 'Tweet']
df = pd.DataFrame(data, columns=columns)

#get latest created_at date from dataframe
latestDate = df['Date'].max()

#write latest date to config file
def writeLatestDate(latestDate):
    with open('config.py', 'r', encoding='utf-8') as file:
        data = file.readlines()
    
    data[6] = 'LATEST_DATE = "' + str(latestDate) + '"'

    with open('config.py', 'w', encoding='utf-8') as file:
        file.writelines(data)

writeLatestDate(latestDate)

#write tweets to db and close connection
def dbWrite(df, conn):
    try:
        cursor = conn.cursor()
        
        for index, row in df.iterrows():
            cursor.execute("INSERT INTO dbo.Twitter (Userkey, Tweet)\
                            Select us.UserId, ?\
                            from Users us\
                            where us.TwitterId = ?", row.Tweet, row.User)
        conn.commit()
        cursor.close()
            
    except pyodbc.Error as e:
        print("Error while connecting to db", e)
        
dbWrite(df, conn)

print("Done")