import pyodbc
import pandas as pd
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer

try:
    conn_str = "Driver={SQL Server};Server=Z690-ELDER;Database=SentimentAnalysis;Trusted_Connection=yes;"
    conn_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(conn_url)
        
    df = pd.read_sql("SELECT * FROM dbo.Twitter", engine)
            
except pyodbc.Error as e:
    print("Error while connecting to db", e)

sw = stopwords.words('english')

#remove useful words from the stopword list
sw.remove('not')
sw.remove('no')
sw.remove('nor')
sw.remove("won't")
sw.remove("wouldn't")
sw.remove("shouldn't")
sw.remove("couldn't")
sw.remove('against')
sw.remove("aren't")
sw.remove("haven't")
sw.remove("hasn't")
sw.remove("doesn't")
sw.remove("isn't")

train_original = pd.read_csv('./TrainingData/trainingdata2.csv')
train_original.columns = ['target','id','date','flag','user','text']

train=train_original[['id','text']]

test = df[['UserKey', 'Tweet']]
test.columns = ['id', 'text']

combine = train.append(test,ignore_index=True,sort=True)

#remove newlines
combine['Tidy_Tweets'] = combine['text'].str.replace("\n"," ")

def remove_pattern(text,pattern):
    
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,text)
    
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)
    
    return text

#change all tweets to lower case
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.lower()
#remove twitter handles (@user)
combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['text'], "@[\w]*")
#remove special characters, numbers, punctuations
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")
#remove short words (length < 3)
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
#remove stopwords
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))
#remove tweets with < 3 words
combine = combine[combine['Tidy_Tweets'].str.split().str.len().gt(3)]

combine.to_csv('data.csv', mode='w')

print("done")
