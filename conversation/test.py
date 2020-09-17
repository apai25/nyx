import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
import pickle
from nltk.corpus import stopwords

test = pd.read_csv('conversation/data/test.csv')
x_test = test.iloc[:, 0].values
y_test = test.iloc[:, 1].values

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

from nltk.stem.porter import PorterStemmer
import re

stemmer = PorterStemmer()
corpus = []
for i in x_test:
	text = re.sub('[^a-zA-Z]', ' ', i)
	text = text.lower()
	text = text.split()
	text = [stemmer.stem(word) for word in text if word not in set(all_stopwords)]
	text = ' '.join(text)
	corpus.append(text)

cv = pickle.load(open('conversation/save/count_vectorizer.pickle', 'rb'))
x_test = cv.transform(x_test)

x_test = x_test.toarray()

model = load_model('conversation/save/model')

predictions = np.argmax(model.predict(x_test), axis=-1)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_test = encoder.fit_transform(y_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')