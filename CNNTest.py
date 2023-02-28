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
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras import backend
import tensorflow as tf
from nltk import word_tokenize
from gensim import models

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

def preprocessTweet(df):
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
test = preprocessTweet(test)

#tokenize
tokens = [word_tokenize(sen) for sen in test['modText']]
test['tokens'] = tokens

all_test_words = [word for tokens in test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))

tokenizer = Tokenizer(num_words=len(TEST_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(test["modText"].tolist())

test_sequences = tokenizer.texts_to_sequences(test["modText"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=50)

#load CNN model
model = tf.keras.models.load_model('CNNModel.h5')

#predict
pred = model.predict(test_cnn_data, batch_size=1024, verbose=1)

labels = [1, 0]

prediction_labels=[]
for p in pred:
    prediction_labels.append(labels[np.argmax(p)])
    
test['sentiment'] = prediction_labels

#save to csv
test.to_csv('CNNResult.csv', index=False)