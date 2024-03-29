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
import pickle

train_original = pd.read_csv('./TrainingData/trainingdata2.csv')
train_original.columns = ['target','id','date','flag','user','text']

test=train_original[['id','text', 'target']]

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

#split test data into test and validation
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(test['modText'])
df_bow = pd.DataFrame(bow.todense())

test_bow = bow[:]
test_bow.todense()

x_test_bow, x_valid_bow, y_test_bow, y_valid_bow = train_test_split(test_bow,test['target'],test_size=0.3,random_state=42)

tfidf = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(test['modText'])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())

test_tfidf_matrix = tfidf_matrix[:]
test_tfidf_matrix.todense()

x_test_tfidf, x_valid_tfidf, y_test_tfidf, y_valid_tfidf = train_test_split(test_tfidf_matrix,test['target'],test_size=0.3,random_state=40)

load_DT_model = pickle.load(open('DT_model.sav', 'rb'))

y_pred_bow = load_DT_model.predict(x_valid_bow)
y_pred_tfidf = load_DT_model.predict(x_valid_tfidf)

#print accuracy, precision, recall, f1-score
acc_bow=accuracy_score(y_valid_bow,y_pred_bow)
prec_bow = precision_score(y_valid_bow, y_pred_bow, average='macro')
rec_bow = recall_score(y_valid_bow, y_pred_bow, average='macro')
f1_bow = f1_score(y_valid_bow, y_pred_bow, average='macro')

print("Accuracy: ", acc_bow)
print("Precision: ", prec_bow)
print("Recall: ", rec_bow)
print("F1-Score: ", f1_bow)

acc_tfidf = accuracy_score(y_valid_tfidf,y_pred_tfidf)
prec_tfidf = precision_score(y_valid_tfidf, y_pred_tfidf, average='macro')
rec_tfidf = recall_score(y_valid_tfidf, y_pred_tfidf, average='macro')
f1_tfidf = f1_score(y_valid_tfidf, y_pred_tfidf, average='macro')

print("Accuracy: ", acc_tfidf)
print("Precision: ", prec_tfidf)
print("Recall: ", rec_tfidf)
print("F1-Score: ", f1_tfidf)

#save confusion matrix to png
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_valid_tfidf, y_pred_tfidf)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells

# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Negative','Positive']); ax.yaxis.set_ticklabels(['Negative','Positive']);

plt.savefig('confusion_matrix_DT2.png')

cm = confusion_matrix(y_valid_bow, y_pred_bow)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells

# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Negative','Positive']); ax.yaxis.set_ticklabels(['Negative','Positive']);

plt.savefig('confusion_matrix_DT1.png')

#plot a chart comparing the accuracy, precision, recall and f1-score of the two models

import matplotlib.pyplot as plt
import numpy as np

#metrics *100
acc_bow = acc_bow*100
prec_bow = prec_bow*100
rec_bow = rec_bow*100
f1_bow = f1_bow*100

acc_tfidf = acc_tfidf*100
prec_tfidf = prec_tfidf*100
rec_tfidf = rec_tfidf*100
f1_tfidf = f1_tfidf*100

# data to plot
n_groups = 4
bow = (acc_bow, prec_bow, rec_bow, f1_bow)
tfidf = (acc_tfidf, prec_tfidf, rec_tfidf, f1_tfidf)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.9

rects1 = plt.bar(index, bow, bar_width,
alpha=opacity,
color='b',
label='Bag of Words')

rects2 = plt.bar(index + bar_width, tfidf, bar_width,
alpha=opacity,
color='g',
label='TF-IDF')

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Scores by metrics and model')
plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall', 'F1-Score'))
plt.legend()

plt.tight_layout()
plt.savefig('DT_scores.png')