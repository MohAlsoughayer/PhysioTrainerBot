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

#-----------------------------------------------------------------------------------------------------------------------------
# NLU: get intent trial 
# Loading json data
# with open('PhysioTrainerBot/data/data_full.json') as file:
#     data = json.loads(file.read())

# # Loading out-of-scope intent data
# val_oos = np.array(data['oos_val'])
# train_oos = np.array(data['oos_train'])
# test_oos = np.array(data['oos_test'])

# # Loading other intents data
# val_others = np.array(data['val'])
# train_others = np.array(data['train'])
# test_others = np.array(data['test'])

# # Merging out-of-scope and other intent data
# val = np.concatenate([val_oos,val_others])
# train = np.concatenate([train_oos,train_others])
# test = np.concatenate([test_oos,test_others])

# # Concatenate all
# data = np.concatenate([train,test,val])
# data = data.T

# # Set text and intent label variables
# text = data[0]
# labels = data[1]

# # Split data
# train_txt,test_txt,train_label,test_labels = train_test_split(text,labels,test_size = 0.3)

# # get padding lengths
# ls=[]
# for c in train_txt:
#     ls.append(len(c.split()))
# maxLen=int(np.percentile(ls, 98))

# # get GloVe Pre-trained word vector to help train the model 
# embeddings_index={}
# with open('PhysioTrainerBot/data/glove.6B.100d.txt', encoding='utf8') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
# all_embs = np.stack(embeddings_index.values())
# emb_mean,emb_std = all_embs.mean(), all_embs.std()

# # Pad and tokenize dataset 
# max_num_words = 40000
# embedding_dim=len(embeddings_index['the'])
# classes = np.unique(labels)

# tokenizer = Tokenizer(num_words=max_num_words)
# tokenizer.fit_on_texts(train_txt)

# train_sequences = tokenizer.texts_to_sequences(train_txt)
# train_sequences = pad_sequences(train_sequences, maxlen=maxLen, padding='post')
# test_sequences = tokenizer.texts_to_sequences(test_txt)
# test_sequences = pad_sequences(test_sequences, maxlen=maxLen, padding='post')
# word_index = tokenizer.word_index

# # get embedded matrix that contains the vector representations of words in our dataset
# num_words = min(max_num_words, len(word_index) )+1
# embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))
# for word, i in word_index.items():
#     if i >= max_num_words:
#         break
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

# # encode labels using OneHotEncoder
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(classes)

# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoder.fit(integer_encoded)

# train_label_encoded = label_encoder.transform(train_label)
# train_label_encoded = train_label_encoded.reshape(len(train_label_encoded), 1)
# train_label = onehot_encoder.transform(train_label_encoded)
# test_labels_encoded = label_encoder.transform(test_labels)
# test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
# test_labels = onehot_encoder.transform(test_labels_encoded)

# # Put the model together 
# model = Sequential()

# model.add(Embedding(num_words, 100, trainable=False,input_length=train_sequences.shape[1], weights=[embedding_matrix]))
# model.add(Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
# model.add(Dropout(0.3))
# model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
# model.add(Dropout(0.3))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(classes.shape[0], activation='softmax'))
# model.summary()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# # Train model 
# history = model.fit(train_sequences, train_label, epochs = 20,
#           batch_size = 64, shuffle=True,
#           validation_data=[test_sequences, test_labels])

# # plot model results
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# # save trained model for future use
# model.save('PhysioTrainerBot/models/intents.h5')

# with open('PhysioTrainerBot/utils/classes.pkl','wb') as file:
#     pickle.dump(classes,file)

# with open('PhysioTrainerBot/utils/tokenizer.pkl','wb') as file:
#     pickle.dump(tokenizer,file)

# with open('PhysioTrainerBot/utils/label_encoder.pkl','wb') as file:
#     pickle.dump(label_encoder,file)

#-----------------------------------------------------------------------------------------------------------------------------
# dataset = load_dataset("AmazonScience/massive", "en-US", split='train')
# define the set of intentions for PT to understand
# ourData = {"intents": [
#             # Add injury, strength, mobility, weight-loss, and stamina
#              {"tag": "injury",
#               "patterns": ["hurts", "hurt", "I feel pain"],
#               "responses": ["I am sorry to hear that, what happened?", "How did that happen?"]
#              },
#              {"tag": "strength",
#               "patterns": ["I want to get bigger", "I want to a sixpack"],
#               "responses": ["That's the spirit", "Look out Dwayne Johnson"]
#              },
#              {"tag": "mobility",
#               "patterns": ["I want to split", "I want to touch my feet"],
#               "responses": ["Soon enough you'll be able to fold yourself like paper"]
#              },
#              {"tag": "weight-loss",
#               "patterns": ["I want a nice body", "I want to be sexy", "beach body", "I want to lose fat"],
#               "responses": ["Beauty is from the inside, but physical health is important"]
#              },
#              {"tag": "stamina",
#               "patterns": ["run a marathon", "not get tired"],
#               "responses": ["Amazing! This will require patience and determination"]
#              },
#              {"tag": "age",
#               "patterns": ["how old are you?"],
#               "responses": ["I am 2 years old and my birthday was yesterday"]
#              },
#               {"tag": "greeting",
#               "patterns": [ "Hi", "Hello", "Hey"],
#               "responses": ["Hi there", "Hello", "Hi :)"],
#              },
#               {"tag": "goodbye",
#               "patterns": [ "bye", "later"],
#               "responses": ["Bye", "take care"]
#              },
#              {"tag": "name",
#               "patterns": ["what's your name?", "who are you?"],
#               "responses": ["My name is PhysioTrainerBot, but you can call me PT for short"]
#              }

# ]}

# lm = WordNetLemmatizer() #for getting words
# # lists
# ourClasses = []
# newWords = []
# documentX = []
# documentY = []
# # Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
# for intent in ourData["intents"]:
#     for pattern in intent["patterns"]:
#         ournewTkns = nltk.word_tokenize(pattern)# tokenize the patterns
#         newWords.extend(ournewTkns)# extends the tokens
#         documentX.append(pattern)
#         documentY.append(intent["tag"])


#     if intent["tag"] not in ourClasses:# add unexisting tags to their respective classes
#         ourClasses.append(intent["tag"])

# newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation] # set words to lowercase if not in punctuation
# newWords = sorted(set(newWords))# sorting words
# ourClasses = sorted(set(ourClasses))# sorting classes

# trainingData = [] # training list array
# outEmpty = [0] * len(ourClasses)
# # bow model
# for idx, doc in enumerate(documentX):
#     bagOfwords = []
#     text = lm.lemmatize(doc.lower())
#     for word in newWords:
#         if word in text:
#             bagOfwords.append(1)
#         else: 
#             bagOfwords.append(0)

#     outputRow = list(outEmpty)
#     outputRow[ourClasses.index(documentY[idx])] = 1
#     trainingData.append([bagOfwords, outputRow])

# random.shuffle(trainingData)
# trainingData = np.array(trainingData, dtype=object)# coverting our data into an array afterv shuffling

# x = np.array(list(trainingData[:, 0]))# first trainig phase
# y = np.array(list(trainingData[:, 1]))# second training phase

# iShape = (len(x[0]),)
# oShape = len(y[0])

# # parameter definition
# ourNewModel = Sequential()
# # In the case of a simple stack of layers, a Sequential model is appropriate

# # Dense function adds an output layer
# ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
# # The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
# ourNewModel.add(Dropout(0.5))
# # Dropout is used to enhance visual perception of input neurons
# ourNewModel.add(Dense(64, activation="relu"))
# ourNewModel.add(Dropout(0.3))
# ourNewModel.add(Dense(oShape, activation = "softmax"))
# # below is a callable that returns the value to be used with no arguments
# md = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
# # Below line improves the numerical stability and pushes the computation of the probability distribution into the categorical crossentropy loss function.
# ourNewModel.compile(loss='categorical_crossentropy',
#               optimizer=md,
#               metrics=["accuracy"])
# # Output the model in summary
# print(ourNewModel.summary())
# # Whilst training your Nural Network, you have the option of making the output verbose or simple.
# ourNewModel.fit(x, y, epochs=200, verbose=1)
# # By epochs, we mean the number of times you repeat a training set.

# def ourText(text):
#   newtkns = nltk.word_tokenize(text)
#   newtkns = [lm.lemmatize(word) for word in newtkns]
#   return newtkns

# def wordBag(text, vocab):
#   newtkns = ourText(text)
#   bagOwords = [0] * len(vocab)
#   for w in newtkns:
#     for idx, word in enumerate(vocab):
#       if word == w:
#         bagOwords[idx] = 1
#   return np.array(bagOwords)

# def Pclass(text, vocab, labels):
#   bagOwords = wordBag(text, vocab)
#   ourResult = ourNewModel.predict(np.array([bagOwords]))[0]
#   newThresh = 0.2
#   yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

#   yp.sort(key=lambda x: x[1], reverse=True)
#   newList = []
#   for r in yp:
#     newList.append(labels[r[0]])
#   return newList

# def getRes(firstlist, fJson):
#   tag = firstlist[0]
#   listOfIntents = fJson["intents"]
#   for i in listOfIntents:
#     if i["tag"] == tag:
#       ourResult = random.choice(i["responses"])
#       break
#   return ourResult

# # Run Chatbot
# chatFlag = True
# while chatFlag:
#     newMessage = input("Me: ")
#     intents = Pclass(newMessage, newWords, ourClasses)
#     ourResult = getRes(intents, ourData)
#     print("PT: " + ourResult)
#     if ourResult in ["Bye", "take care"]:
#         chatFlag = False
