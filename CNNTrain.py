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
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential

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

def preprocessTweet(df):
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
    #remove quotes
    df['text'] = df['text'].str.replace("quot", "")
    #remove stopwords
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (sw)]))
    #remove short words (length < 2)
    df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if (len(w)>2)]))
    #remove duplicate tweets - bot prevention
    df['text'] = df['text'].drop_duplicates(keep=False)
    #remove NANs
    df.dropna(inplace=True)
    #remove empty tweets
    df = df[df.text != '']
    return df

train = preprocessTweet(train)

#split into train and test while splitting user tweets equally
train, test = train_test_split(train, test_size=0.2, random_state=42, stratify=train['sentiment'])

#define vocabulary
vocab = Counter()

#add all words from the training set to the vocabulary
for tweet in train['text']:
    for word in tweet.split():
        vocab[word] += 1

#print the size of the vocabulary
print(len(vocab))

#print the 20 most common words in the vocabulary
print(vocab.most_common(20))

#keep only the words that appear more than 2 times
words = [word for word,count in vocab.items() if count >= 2]

#filter out the words that are not in the vocabulary
train['text'] = train['text'].apply(lambda x: ' '.join([word for word in x.split() if word in (words)]))

#tokenize the tweets
tokenize = Tokenizer()
tokenize.fit_on_texts(train['text'])

#sequence encode the tweets
encoded_train = tokenize.texts_to_sequences(train['text'])
encoded_test = tokenize.texts_to_sequences(test['text'])

#pad sequences for efficeint processing
max_length = max([len(s.split()) for s in train['text']])
X_train = pad_sequences(encoded_train, maxlen=max_length, padding='post')
Y_train = train['sentiment']

X_test = pad_sequences(encoded_test, maxlen=max_length, padding='post')
Y_test = test['sentiment']

vocab_size = len(tokenize.word_index) + 1

#define CNN model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(64, 8, activation='relu'))
model.add(MaxPooling1D(2))
model.add(flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

#compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, verbose=2)

#evaluate model
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))