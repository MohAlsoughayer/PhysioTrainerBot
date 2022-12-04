# Author: Mohammed Alsoughayer (mohammed.alsoughayer@gmail.com)
# Description: Trial to Implement a Basic Chatbot (This implimentation was done by following along 'https://www.analyticsvidhya.com/blog/2021/10/complete-guide-to-build-your-ai-chatbot-with-nlp-in-python/')

# Get necessary packages 
import os
import datetime
import time
import numpy as np
import transformers
import tensorflow as tf
import tensorflow_datasets as tfds

# Building the AI
class ChatBot():
    def __init__(self, name):
        print("----- Starting up", name, "-----")
        self.name = name

    def speech_to_text(self):
        self.text = input("Me  --> ")
            

    @staticmethod
    def text_to_speech(text):
        print("PhysioTrainerBot --> ", text)
        
        

    def wake_up(self, text):
        return True if self.name.lower() in text.lower() else False

    @staticmethod
    def action_time():
        return datetime.datetime.now().time().strftime('%H:%M')


# Running the AI
if __name__ == "__main__":
    
    ai = ChatBot(name="PT")
    nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    ex=True
    while ex:
        ai.speech_to_text()

        ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am PhysioTrainerBot the AI, but you can call me PT for short, what can I do for you?"
        
        ## action time
        elif "time" in ai.text:
            res = ai.action_time()
        
        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","mention not"])
        
        elif any(i in ai.text for i in ["exit","close"]):
            res = np.random.choice(["Tata","Have a good day","Bye","Goodbye","Hope to meet soon","peace out!"])
            
            ex=False
        ## conversation
        else:   
            if ai.text=="ERROR":
                res="Sorry, come again?"
            else:
                chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >> ")+6:].strip()

        ai.text_to_speech(res)
    print("----- Closing down PhysioTrainerBot -----")
