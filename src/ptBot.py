# Author: Mohammed Alsoughayer
# Descritpion: PhysionTrainerBot implimentation

# Import necessary packages
import numpy as np
import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import pickle
import string
import random
import nltk
from nltk.stem import WordNetLemmatizer # It has the ability to lemmatize.
from keras import Sequential # Sequential groups a linear stack of layers into a tf.keras.Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional,Embedding
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

# NLU 
# load model and accompanying args
# model = load_model('PhysioTrainerBot/models/intents.h5')

# with open('PhysioTrainerBot/utils/classes.pkl','rb') as file:
#   classes = pickle.load(file)

# with open('PhysioTrainerBot/utils/tokenizer.pkl','rb') as file:
#   tokenizer = pickle.load(file)

# with open('PhysioTrainerBot/utils/label_encoder.pkl','rb') as file:
#   label_encoder = pickle.load(file)

# # create class
# class IntentClassifier:
#     def __init__(self,classes,model,tokenizer,label_encoder):
#         self.classes = classes
#         self.classifier = model
#         self.tokenizer = tokenizer
#         self.label_encoder = label_encoder

#     def get_intent(self,text):
#         self.text = [text]
#         self.test_keras = self.tokenizer.texts_to_sequences(self.text)
#         self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
#         self.pred = self.classifier.predict(self.test_keras_sequence)
#         return label_encoder.inverse_transform(np.argmax(self.pred,1))[0]

# # Initialize nlu object 
# nlu = IntentClassifier(classes,model,tokenizer,label_encoder)
# # NLG

# # Run PTBot
# botFlag = True
# while botFlag:
#     inp = input('Me: ')
#     if inp == 'close':
#         print('closing....')
#         botFlag = False
#     else: 
#         print("intent: " + nlu.get_intent(inp))

# Building the AI
class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name

    def get_input(self):
        self.text = input("Me  --> ")
            

    @staticmethod
    def respond(text):
        print("PTBot --> ", text)
        
        

    def wake_up(self, text):
        return True if self.name.lower() in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')