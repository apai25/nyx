import numpy as np 
import pandas as pd 
import tensorflow as tf
import nltk
import pickle
from nltk.corpus import stopwords

train = pd.read_csv('conversation/data/train.csv')
x_train = train.iloc[:, 0].values
y_train = train.iloc[:, 1:2].values

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

from nltk.stem.porter import PorterStemmer
import re

stemmer = PorterStemmer()
corpus = []
for i in x_train:
	text = re.sub('[^a-zA-Z]', ' ', i)
	text = text.lower()
	text = text.split()
	text = [stemmer.stem(word) for word in text if word not in set(all_stopwords)]
	text = ' '.join(text)
	corpus.append(text)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(corpus)
pickle.dump(cv, open('conversation/save/count_vectorizer.pickle', 'wb'))

x_train = x_train.toarray()

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
encoder = OneHotEncoder(dtype=int)
ct = ColumnTransformer([('encoder', encoder, [0])], remainder='passthrough')
y_train = ct.fit_transform(y_train)

y_train = y_train.toarray()

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Dense(units=64, activation='relu', input_shape=(x_train.shape[0], x_train.shape[1])))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=11, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=1000, batch_size=32)

model.save('conversation/save/model', save_format='tf')