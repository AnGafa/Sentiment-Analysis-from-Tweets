import pickle
import re
import string
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyodbc
import seaborn as sns
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn import preprocessing, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

train_original = pd.read_csv('./TrainingData/trainingdata2.csv')
train_original.columns = ['target','id','date','flag','user','text']

train=train_original[['id','text', 'target']]

del train_original
 
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
    df['text'] = df['text'].str.replace("\n"," ")
    #turn all text to lowercase
    df['text'] = df['text'].str.lower()
    # remove twitter handles (@user)
    df['text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")
    #remove links
    df['text'] = df['text'].str.replace('http\S+|www.\S+', '', case=False)
    #remove special characters, numbers, punctuations
    df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")
    #remove short words (length < 3)
    df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if (len(w)>3 or w == 'no')]))
    #remove duplicate tweets - bot prevention
    df['text'] = df['text'].drop_duplicates(keep=False)
    #remove quotes
    df['text'] = df['text'].str.replace("quot", "")
    #remove NANs
    df.dropna(inplace=True)
    #remove stopwords
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))
    #remove empty tweets
    df = df[df.text != '']
    #stemming
    df['text'] = df['text'].apply(stem_sentences)
    return df

train = preprocessTweet(train, sw)

#decision tree classifier
x_trainFull, y_trainFull = train.text.values, train['target']

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(train['text'])
df_bow = pd.DataFrame(bow.todense())

train_bow = bow[:]
train_bow.todense()

#split train data into train and validation
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow,train['target'],test_size=0.2,random_state=42)

#decision trees hyperparameters
criterion = ['gini', 'entropy']
splitter = ['best', 'random']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
max_features = ['sqrt', 'log2']

hyperparameters = dict(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)

decision_tree = tree.DecisionTreeClassifier()
clf = GridSearchCV(decision_tree, hyperparameters, cv=10, verbose=2, n_jobs=11)
clf.fit(x_train_bow, y_train_bow)
y_pred = clf.predict(x_valid_bow)
 
acc=accuracy_score(y_valid_bow, y_pred)
precision=precision_score(y_valid_bow, y_pred)
recall=recall_score(y_valid_bow, y_pred)
f1=f1_score(y_valid_bow, y_pred)

print("Accuracy: ", acc)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)

#best hyperparameters
print('Best Criterion:', clf.best_estimator_.get_params()['criterion'])
print('Best Splitter:', clf.best_estimator_.get_params()['splitter'])
print('Best Max Depth:', clf.best_estimator_.get_params()['max_depth'])
print('Best Min Samples Split:', clf.best_estimator_.get_params()['min_samples_split'])
print('Best Min Samples Leaf:', clf.best_estimator_.get_params()['min_samples_leaf'])
print('Best Max Features:', clf.best_estimator_.get_params()['max_features'])

#save model
#filename = 'DT_model.sav'
#pickle.dump(clf, open(filename, 'wb'))

print("done")