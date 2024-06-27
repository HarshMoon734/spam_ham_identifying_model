import pandas as pd
import numpy as np

import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

__dirname = r"C:\Users\Harsh\Downloads\fake_news\train.csv"

df = pd.read_csv(__dirname)
df = df.dropna().reset_index(drop=True)
df = df.sample(2000).reset_index(drop = True)

def remove_punctuation(text):
    
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

stemmer = PorterStemmer()

df["content"] = df["author"]+" "+df["title"]+" "+df["text"]

df = df.drop(["author","title","text"], axis=1)

conten = []
for i in range(0,df.shape[0]):
    conten.append(stemmer.stem(remove_punctuation(df["content"][i].lower())))

df["content"] = conten

vectorizer = CountVectorizer()

x = vectorizer.fit_transform(df["content"]).toarray()
y = df["label"]

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7)
model = LogisticRegression(max_iter=2000)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))