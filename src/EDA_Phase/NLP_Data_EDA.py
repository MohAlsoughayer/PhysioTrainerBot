# Author: Mohammed Alsoughayer
# Descritpion: Perform EDA on the gathered data to narrow down the NLP Datasets for the chatbot 

# Import necessary packages
import numpy as np
import pandas as pd
import math
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
#-----------------------------------------------------------------------------------------------------------------------------
# dataset = load_dataset("AmazonScience/massive", "en-US", split='train')
# define the set of intentions for PT to understand
ourData = {"intents": [
             {"tag": "injury",
              "patterns": ["hurts", "hurt", "I feel pain", "break"],
              "responses": ["I am sorry to hear that, what happened?", "How did that happen?"]
             },
             {"tag": "strength",
              "patterns": ["I want to get bigger", "I want to a sixpack"],
              "responses": ["That's the spirit", "Look out Dwayne Johnson"]
             },
             {"tag": "mobility",
              "patterns": ["I want to split", "I want to touch my feet"],
              "responses": ["Soon enough you'll be able to fold yourself like paper"]
             },
             {"tag": "weight-loss",
              "patterns": ["I want a nice body", "I want to be sexy", "beach body", "I want to lose fat"],
              "responses": ["With consistency, You'll look fitter than ever."]
             },
             {"tag": "stamina",
              "patterns": ["run a marathon", "not get tired"],
              "responses": ["Amazing! This will require patience and determination"]
             },
              {"tag": "greeting",
              "patterns": [ "Hi", "Hello", "Hey"],
              "responses": ["Hi there", "Hello", "Hi :)"],
             },
              {"tag": "goodbye",
              "patterns": [ "bye", "later"],
              "responses": ["Bye", "take care"]
             },
             {"tag": "name",
              "patterns": ["what's your name?", "who are you?"],
              "responses": ["My name is PhysioTrainerBot, but you can call me PT for short"]
             }
]}

lm = WordNetLemmatizer() #for getting words
# lists
ourClasses = []
newWords = []
documentX = []
documentY = []
# Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
for intent in ourData["intents"]:
    for pattern in intent["patterns"]:
        ournewTkns = nltk.word_tokenize(pattern)# tokenize the patterns
        newWords.extend(ournewTkns)# extends the tokens
        documentX.append(pattern)
        documentY.append(intent["tag"])


    if intent["tag"] not in ourClasses:# add unexisting tags to their respective classes
        ourClasses.append(intent["tag"])

newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation] # set words to lowercase if not in punctuation
newWords = sorted(set(newWords))# sorting words
ourClasses = sorted(set(ourClasses))# sorting classes

trainingData = [] # training list array
outEmpty = [0] * len(ourClasses)
# bow model
for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        if word in text:
            bagOfwords.append(1)
        else: 
            bagOfwords.append(0)

    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

random.shuffle(trainingData)
trainingData = np.array(trainingData, dtype=object)# coverting our data into an array afterv shuffling

x = np.array(list(trainingData[:, 0]))# first trainig phase
y = np.array(list(trainingData[:, 1]))# second training phase

iShape = (len(x[0]),)
oShape = len(y[0])

# parameter definition
ourNewModel = Sequential()
# In the case of a simple stack of layers, a Sequential model is appropriate

# Dense function adds an output layer
ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
# The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
ourNewModel.add(Dropout(0.5))
# Dropout is used to enhance visual perception of input neurons
ourNewModel.add(Dense(64, activation="relu"))
ourNewModel.add(Dropout(0.3))
ourNewModel.add(Dense(oShape, activation = "softmax"))
# below is a callable that returns the value to be used with no arguments
md = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
# Below line improves the numerical stability and pushes the computation of the probability distribution into the categorical crossentropy loss function.
ourNewModel.compile(loss='categorical_crossentropy',
              optimizer=md,
              metrics=["accuracy"])
# Output the model in summary
print(ourNewModel.summary())
# Whilst training your Nural Network, you have the option of making the output verbose or simple.
ourNewModel.fit(x, y, epochs=200, verbose=1)
# By epochs, we mean the number of times you repeat a training set.

def ourText(text):
  newtkns = nltk.word_tokenize(text)
  newtkns = [lm.lemmatize(word) for word in newtkns]
  return newtkns

def wordBag(text, vocab):
  newtkns = ourText(text)
  bagOwords = [0] * len(vocab)
  for w in newtkns:
    for idx, word in enumerate(vocab):
      if word == w:
        bagOwords[idx] = 1
  return np.array(bagOwords)

def Pclass(text, vocab, labels):
  bagOwords = wordBag(text, vocab)
  ourResult = ourNewModel.predict(np.array([bagOwords]))[0]
  newThresh = 0.2
  yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

  yp.sort(key=lambda x: x[1], reverse=True)
  newList = []
  for r in yp:
    newList.append(labels[r[0]])
  return newList

def getRes(firstlist, fJson):
  tag = firstlist[0]
  listOfIntents = fJson["intents"]
  for i in listOfIntents:
    if i["tag"] == tag:
      ourResult = random.choice(i["responses"])
      break
  return ourResult


# create necessary objects 
chatFlag = True
injuredFlag = False
greeting = "PT: Hello my name is PhysioTrainerBot (PT for short), I'm a physical health assistant.\n How can I help you today?"
general_message = '''The most important insight/recommendation I can give you is that taking care of your physical health requires consistency.
Therefore, do not push your limits too far and try to get the most amount pleasure in your exercises.
If you're having a rough day and the idea of working out is dreadful, then just do the bare minimum  to give you the satisfaction of showing up.'''
flex_link = 'https://www.self.com/gallery/essential-stretches-slideshow'
# Run Chatbot
print(greeting)
while chatFlag:
    newMessage = input("Me: ")
    intents = Pclass(newMessage, newWords, ourClasses)
    ourResult = getRes(intents, ourData)
    print(f"PT: {ourResult}")
    if ourResult in ["I am sorry to hear that, what happened?", "How did that happen?"]: #injured
        injuredFlag = True 
    elif ourResult in ["That's the spirit", "Look out Dwayne Johnson"]: # strength
        # create nonInjured object (get goal/location)
    elif ourResult in ["Soon enough you'll be able to fold yourself like paper"]: # flexibility
        print(f'Here\'s a link I have for general flexibility exercises: {flex_link}')
    elif ourResult in ["With consistency, You'll look fitter than ever."]: # weigth loss
        # create nonInjured object (general goal)
    elif ourResult in ["Amazing! This will require patience and determination"]: # stamina 
        # create nonInjured object (stamina goal)
    elif ourResult in ["Bye", "take care"]: # close
        chatFlag = False
#-----------------------------------------------------------------------------------------------------------------------------
