# PhysioTrainerBot

## Introduction 
Almost every business has some service to provide its customers. Nowadays, most of these services can be offered through the companies’ websites/applications. However, navigating through these platforms to find the service you desire can be tedious and time-consuming. Therefore, having a chat bot on the company website and/or applications can help these users to allocate the desired service, or provide clarification. 

Moreover, physical health is a common goal amongst all humans; whether we are injured or just simply aren’t as healthy as we want, having a well established exercise routine will go a long way in our physical and mental status. However, putting together that routine can be very cumbersome to some people and might halt their motivation to become healthier. Also, going to a physical therapist or getting a personal trainer to help put together a routine might be too expensive for some people. 

That being said, the goal for this project is to build a web-app consisting of chatbot to converse with any user. The purpose of the chatbot is to serve as a free physical therapist/trainer. It would converse with the user to understand their physical situation and goal. Then It will recommend a training plan to aid them reach the physical health they desire. Disclaimer: The chatbot analysis for injured users shouldn’t be taken as a diagnosis, rather, it should serve as an educational tool to help the user in understanding the possible condition of their health and the exercises that could aid their physical health recovery.
## Current Version Data Sources
As of right now, PhysioTrainerBot uses NLP techniques to identify the intent of the user from the following options: strength, stamina, or weight-loss. Then depending on their goal it will generate 7-day routine. 
### NLP Dataset 
The dataset used to train the Nural-net model (manually entered) is shown bellow:
```
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
              "patterns": ["I want to split", "I want to touch my feet", "I want to be flexible"],
              "responses": ["Soon enough you'll be able to fold yourself like paper"]
             },
             {"tag": "weight-loss",
              "patterns": ["I want a nice body", "I want to be sexy", "beach body", "I want to lose fat", "I want to get fit"],
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
              "patterns": [ "bye", "later", "thanks"],
              "responses": ["Bye", "take care"]
             },
             {"tag": "name",
              "patterns": ["what's your name?", "who are you?"],
              "responses": ["My name is PhysioTrainerBot, but you can call me PT for short"]
             }
]}
```
### Exercises Dataset
To get a workouts dataset, I scraped the table from the following link:
https://en.wikipedia.org/wiki/List_of_weight_training_exercises
## References 
Here is a list of references I used to complete the current version of the project:
http://www.ylz.ncx.mybluehost.me/scavetta/misk-dsi-2022/08_DL/_book/
https://realpython.com/python-web-scraping-practical-introduction/
https://research.aimultiple.com/chatbot-architecture/
https://research.aimultiple.com/types-of-conversational-ai/
https://research.aimultiple.com/chatbot-best-practices/
https://www.section.io/engineering-education/creating-chatbot-using-natural-language-processing-in-python/
https://realpython.com/python-keras-text-classification/
https://medium.com/analytics-vidhya/creating-your-own-intent-classifier-b86e000a4926
https://www.physio-pedia.com/home/#
https://en.wikipedia.org/wiki/List_of_weight_training_exercises
https://machinelearningmastery.com/web-crawling-in-python/
https://ocw.mit.edu/courses/6-864-advanced-natural-language-processing-fall-2005/pages/lecture-notes/
https://stackoverflow.com/
