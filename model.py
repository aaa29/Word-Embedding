#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 2018

@author: amine

Word2Vec
"""

from data_utils import clean_str2
import pandas as pd
import numpy as np
import scipy as sp
import tensorflow as tf




from keras.preprocessing.sequence import skipgrams
from keras.layers import Dense, Embedding, Reshape, Merge
from keras.models import Sequential
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split








def gen_batch(skip_grams):
    
    for sg in skip_grams:
        target_words = np.array(list(zip(*sg[0]))[0], dtype='int32')
        context_words = np.array(list(zip(*sg[0]))[1], dtype='int32')
        labels = np.array(sg[1], dtype='int32')

        
        yield target_words, context_words, labels



#data preparation
#reading and cleaning:
        
data = pd.read_csv('datasets/MOV.csv')

data['text'] = data['text'].apply(clean_str2)

texts = [sent[:10] for sents in data['text'].tolist()[:100]
                   for sent in sents]


vocabulary = set([word for text in texts for word in text])

vocab_size = len(vocabulary)



wordToNum = {word : word_id for word_id, word in enumerate(vocabulary)}

NumToWord = {word_id : word for word, word_id in wordToNum.items()}



#convert words to num in text
texts_transformed = [[wordToNum[word] for word in text] for text in texts]


skip_grams = [skipgrams(para, vocabulary_size = vocab_size, window_size = 2) for para in texts_transformed]



#learning parameters

embed_size = 200
epochs = 30

    

model_1 = Sequential()
model_1.add(Embedding(vocab_size, embed_size, input_length = 1))
model_1.add(Reshape((embed_size, )))

model_2 = Sequential()
model_2.add(Embedding(vocab_size, embed_size, input_length = 1))
model_2.add(Reshape((embed_size, )))


model = Sequential()
model.add(Merge([model_1, model_2], mode="dot"))
model.add(Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")





for epoch in range(epochs):
    
    loss =0
    
    batch_generator = gen_batch(skip_grams)
    
    for _ in skip_grams:
        
        batch_X1, batch_X2, batch_Y = next(batch_generator)
        
        batch_X = [batch_X1, batch_X2]
        
        loss += model.train_on_batch(batch_X, batch_Y)
        
    print('Epoch:', epoch, 'Loss:', loss)
    
    

        
        
        






