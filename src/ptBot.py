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
from nltk.stem import WordNetLemmatizer 
from keras import Sequential 
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional,Embedding
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

# load model 

# create chat object 

# run chatbot 
# chatFlag = True
# injuredFlag = False
# greeting = "PT: Hello my name is PhysioTrainerBot (PT for short), I'm a physical health assistant.\n How can I help you today?"
# print(greeting)
# while chatFlag:
#     newMessage = input("Me: ")
#     intents = Pclass(newMessage, newWords, ourClasses)
#     ourResult = getRes(intents, ourData)
#     print(f"PT: {ourResult}")
#     if ourResult in ["I am sorry to hear that, what happened?", "How did that happen?"]: #injured
#         injuredFlag = True 
#     elif ourResult in ["That's the spirit", "Look out Dwayne Johnson"]: # strength
#     elif ourResult in ["Soon enough you'll be able to fold yourself like paper"]: # flexibility
#     elif ourResult in ["Beauty is from the inside, but physical health is important"]: # weigth loss
#     elif ourResult in ["Amazing! This will require patience and determination"]: # stamina 
#     elif ourResult in ["Bye", "take care"]: # close
#         chatFlag = False