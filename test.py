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
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import pickle
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim.similarities import SoftCosineSimilarity

df = pd.read_csv('./results4.csv')
df.columns = ['id', 'text', 'modText', 'sentiment', 'topics', 'hashtags']

dndf = pd.DataFrame(columns=['id', 'text', 'modText', 'sentiment', 'topics', 'hashtags'])

negatives = ['not', 'no', 'nor', "won't", "wouldn't", "shouldn't", "couldn't", 'against', "aren't", "haven't", "hasn't", "doesn't", "isn't", "don't"]

#tokenize text and lowercase
df['tokText'] = df['text'].apply(lambda x: x.split())
df['tokText'] = df['tokText'].apply(lambda x: [y.lower() for y in x])

for i in range(0, len(df)):
    df['topics'][i] = df['topics'][i].replace("'',", "")
df['topics'] = df['topics'].apply(lambda x: x[1:-1].split(', '))

#if text contains more than 1 negative word, add to dndf
for index, row in df.iterrows():
    if len([x for x in negatives if x in row['tokText']]) > 1:
        dndf = dndf.append(row)

#from a selected text, find the most similar users and predict sentiment towards the text based on the topic
selectedIndex = 19
selectedIndex = df[df['text'] == dndf.iloc[selectedIndex]['text']].index[0]

userDF = pd.DataFrame(columns=['id', 'topics',])
userDF.id = df.id.unique()
#add all unique topics to userDF by id unless it is empty
for i in range(0, len(userDF)):
    userDF['topics'][i] = df[df['id'] == userDF['id'][i]]['topics'].sum()
    userDF['topics'][i] = [x for x in userDF['topics'][i] if x != "\'\'"]
    
userDF['topics'].to_csv('./test.csv', index=False)

#change sentiment 4 to 1
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 4 else x)

#for each topic add sentiment based on mean of sentiment of the user's tweets with that topic
for i in range(0, len(userDF)):
    for j in range(0, len(userDF['topics'][i])):
        userDF['topics'][i][j] = userDF['topics'][i][j] + str((df[(df['id'] == userDF['id'][i]) & (df['topics'].apply(lambda x: userDF['topics'][i][j] in x))]['sentiment'].mean()).round(2))

userDF['topicsRaw'] = pd.read_csv('./test.csv')['topics']

#convert array of topicsRaw to string
userDF['topicsRaw'] = userDF['topicsRaw'].apply(lambda x: str(x))

#clean topicsRaw and convert to array
userDF['topicsRaw'] = userDF['topicsRaw'].apply(lambda x: x.replace("\"", ""))
userDF['topicsRaw'] = userDF['topicsRaw'].apply(lambda x: x[1:-1].split(', '))

#using soft cosine similarity return 10 most similar users to a given user
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
documents = userDF['topics']
dictionary = Dictionary(documents)
bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
index = WordEmbeddingSimilarityIndex(fasttext_model300)
sims = SparseTermSimilarityMatrix(index, dictionary)
model = SoftCosineSimilarity(bow_corpus, sims, num_best=10)

def get_recommendations(id):
    idx = userDF.index[userDF['id'] == id][0]
    query = userDF['topics'][idx]
    query = dictionary.doc2bow(query)
    return model[query]

#get closest users to the given user to predict sentiment towards a topic
results = get_recommendations(df['id'][selectedIndex])

#only use users with a distance of less than 0.3
results = [x for x in results if x[1] < 0.9]

#assign user{number} to each user
userDF['user'] = ["user" + str(i) for i in range(0, len(userDF))]
userDF['userNum'] = [str(i) for i in range(0, len(userDF))]

print("Closest users to", userDF.loc[userDF['id'] == df['id'][selectedIndex], 'user'].iloc[0], "are:", [userDF['user'][x[0]] for x in results], "with distances of:", [x[1] for x in results])

#predict sentiment towards the chosen text based on the closest users' sentiment towards the topic
topics = df['topics'][selectedIndex]

#find texts from close users with at least 1 topic in common with the chosen text and add to df
closeDF = pd.DataFrame(columns=['id', 'text', 'modText', 'sentiment', 'topics', 'hashtags'])
for i in range(0, len(results)):
    closeDF = closeDF.append(df[(df['id'] == userDF['id'][results[i][0]]) & (df['topics'].apply(lambda x: any(item in x for item in topics)))])

#perform knn on the close users' sentiment towards the topic
from sklearn.neighbors import KNeighborsClassifier

#encode topics with bag of words
from sklearn.feature_extraction.text import CountVectorizer

#change topics to string
closeDF['topics'] = closeDF['topics'].astype(str)
closeDF['topics'] = closeDF['topics'].astype(pd.StringDtype())
for i in range(0, len(closeDF)):
    print("1")
    closeDF['topics'][i] = closeDF['topics'][i].replace("'", "")
    closeDF['topics'][i] = closeDF['topics'][i].replace(",", "")
    closeDF['topics'][i] = closeDF['topics'][i].replace("[", "")
    closeDF['topics'][i] = closeDF['topics'][i].replace("]", "")
    closeDF['topics'][i] = closeDF['topics'][i].replace("\"", "")

vectorizer = CountVectorizer()
X = vectorizer.fit(closeDF['topics'])

#train knn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, closeDF['sentiment'])

#predict sentiment towards the chosen text
print("Predicted sentiment towards", dndf.iloc[selectedIndex]['text'], "is", knn.predict([[df['id'][selectedIndex], df['text'][selectedIndex], df['modText'][selectedIndex], df['topics'][selectedIndex], df['hashtags'][selectedIndex]]]))