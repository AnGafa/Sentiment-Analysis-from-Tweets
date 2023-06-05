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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import pickle

try:
    conn_str = "Driver={SQL Server};Server=Z690-ELDER;Database=SentimentAnalysis;Trusted_Connection=yes;"
    conn_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(conn_url)
        
    df = pd.read_sql("SELECT * FROM dbo.Twitter", engine)
            
except pyodbc.Error as e:
    print("Error while connecting to db", e)


test = df[['UserKey', 'Tweet']]
test.columns = ['id', 'text']
 
#region prepare stopwords list
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
#endregion

def remove_pattern(text,pattern):
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,text)
    
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)
    
    return text

def stem_sentences(sentence):
    #tokenize the sentence and remove the stems of the words
    ps = PorterStemmer()
    tokens = sentence.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def preprocessTweet(df, sw):
    #remove newlines
    df['modText'] = df['modText'].str.replace("\n"," ")
    #turn all text to lowercase
    df['modText'] = df['modText'].str.lower()
    # remove twitter handles (@user)
    df['modText'] = np.vectorize(remove_pattern)(df['modText'], "@[\w]*")
    #remove links
    df['modText'] = df['modText'].str.replace('http\S+|www.\S+', '', case=False)
    #remove special characters, numbers, punctuations
    df['modText'] = df['modText'].str.replace("[^a-zA-Z#]", " ")
    #remove short words (length < 3)
    df['modText'] = df['modText'].apply(lambda x: ' '.join([w for w in x.split() if (len(w)>3 or w == 'no')]))
    #remove duplicate tweets - bot prevention
    df['modText'] = df['modText'].drop_duplicates(keep=False)
    #remove quotes
    df['modText'] = df['modText'].str.replace("quot", "")
    #remove NANs
    df.dropna(inplace=True)
    #remove stopwords
    df['modText'] = df['modText'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))
    #remove empty tweets
    df = df[df.modText != '']
    #stemming
    df['modText'] = df['modText'].apply(stem_sentences)
    return df

test['modText'] = test['text']
test = preprocessTweet(test, sw)

#decision tree classifier
x_test = test.modText.values

#encode target
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow_test = bow_vectorizer.fit_transform(x_test)
df_test_bow = pd.DataFrame(bow_test.todense())

test_bow = bow_test[:]
test_bow.todense()

y_test = test['id']

load_DT_model = pickle.load(open('DT_model.sav', 'rb'))

y_pred = load_DT_model.predict(test_bow)

#add sentiment to test dataframe
test['sentiment'] = y_pred

#drop modText column
#test = test.drop('modText', axis=1)

#save to csv
#test.to_csv('results.csv', index=False)

print("done")