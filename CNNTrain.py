import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyodbc
import seaborn as sns
import tensorflow as tf
from gensim import models
from keras import backend
from keras.layers import (Conv1D, Dense, Dropout, Embedding, Flatten,
                          GlobalMaxPooling1D, Input, Reshape, concatenate)
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

train_original = pd.read_csv('./TrainingData/trainingdata2.csv')
train_original.columns = ['target','id','date','flag','user','text']

df=train_original[['id','text', 'target']]

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

df = preprocessTweet(df)

#tokenize
tokens = [word_tokenize(sen) for sen in df['text']]
df['tokens'] = tokens

#change target from 4 to 1
df['target'] = df['target'].replace(4,1)

pos = []
neg = []
for l in df.target:
    if l == 0:
        pos.append(0)
        neg.append(1)
    elif l == 1:
        pos.append(1)
        neg.append(0)
df['Pos']= pos
df['Neg']= neg

#split into train and test while splitting user tweets equally
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

all_training_words = [word for tokens in train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in test["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))

word2vec_path = 'GoogleNews-vectors-negative300.bin'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)

training_embeddings = get_word2vec_embeddings(word2vec, train, generate_missing=True)
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300

tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(train["text"].tolist())
training_sequences = tokenizer.texts_to_sequences(train["text"].tolist())

train_word_index = tokenizer.word_index

train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)

test_sequences = tokenizer.texts_to_sequences(test["text"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=False)
    
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2,3,4,5,6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)


    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)  
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['acc'])
    model.summary()
    return model

label_names = ['Pos', 'Neg']

y_train = train[label_names].values

x_train = train_cnn_data
y_tr = y_train

model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
len(list(label_names)))

num_epochs = 14
batch_size = 34

hist = model.fit(x_train, y_tr, epochs=num_epochs, validation_split=0.1, shuffle=True, batch_size=batch_size)

predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)

labels = [1, 0]

prediction_labels=[]
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])
    
print(sum(test['target']==prediction_labels)/len(prediction_labels))

#plot accuracy and loss
import matplotlib.pyplot as plt

#start epoch from 1
'''hist.history['acc'].insert(0,0)
hist.history['loss'].insert(0,0)
hist.history['val_acc'].insert(0,0)
hist.history['val_loss'].insert(0,0)'''

plt.style.use('ggplot')
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.savefig('CNNmodel2000.png')

import matplotlib.pyplot as plt
import seaborn as sns
#plot confusion matrix and save it
from sklearn.metrics import confusion_matrix

plt.figure(figsize=(12,8))
cm = confusion_matrix(test['target'], prediction_labels)
ax= plt.subplot()
hm = sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells
fig = hm.get_figure()
fig.savefig('CNNCM2000.png')

#save model
model.save('CNNmodel2000.h5')