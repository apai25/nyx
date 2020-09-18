import numpy as np 
import pandas as pd 
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def predict(text):
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    cv = pickle.load(open('conversation/save/count_vectorizer.pickle', 'rb'))
    ps = PorterStemmer()
    model = load_model('conversation/save/model')

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in set(all_stopwords)]
    text = [' '.join(text)]
    text = cv.transform(text)
    text = text.toarray()
    prediction = np.argmax(model.predict(text), axis=-1)
    return prediction[0]

text = str(input('Enter string:'))
print(predict(text))