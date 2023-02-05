import pandas as pd
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models
import gensim
import gensim.corpora as corpora
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models import CoherenceModel
from nltk import PorterStemmer
import requests

df = pd.read_csv('./results3.csv')

for i in range(0, len(df)):
    df['topics'][i] = df['topics'][i].replace("\"", "")
    df['topics'][i] = df['topics'][i].replace("\\", "")
    df['topics'][i] = df['topics'][i].replace("'''", "'")
    df['topics'][i] = df['topics'][i].replace("' ", "'")
    df['topics'][i] = df['topics'][i].replace("\",", "'")
    df['topics'][i] = df['topics'][i].replace("'''", "'")
    df['topics'][i] = df['topics'][i].replace("''", "'")
    df['topics'][i] = df['topics'][i].replace("[', ", "[")
    df['topics'][i] = df['topics'][i].replace("'", "")
df['topics'] = df['topics'].apply(lambda x: x[1:-1].split(', '))

api_key = "77307d4d-4d53-4f6b-a7b7-ca1363ed9ed7"
url = "https://api.oneai.com/api/v0/pipeline"
headers = {
  "api-key": api_key, 
  "content-type": "application/json"
}

try:
    for i in range(0, len(df)):
        print(i)
        if(df['topics'][i] == ['']):
            payload = {
                "input": df['text'][i],
                "input_type": "article",
                "output_type": "json",
                "multilingual": {
                "enabled": True
                },
                "steps": [
                    {
                    "skill": "article-topics"
                }
                ],
            }
            r = requests.post(url, json=payload, headers=headers)
            data = r.json()
            for a in data['output']:
                for b in a['labels']:
                    df['topics'][i].append(b['value'])
except Exception as e:
    print(e)
    pass

#hashtag extraction
#df['hashtags'] = df['text'].apply(lambda x: re.findall(r'#(\w+)', x))

df.to_csv('./results4.csv', index=False)