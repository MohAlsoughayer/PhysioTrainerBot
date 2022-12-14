# Author: Mohammed Alsoughayer
# Descritpion: chatbot non injured users module

# import necessary packages
# Data Packages
import csv
import numpy as np
import pandas as pd

# Import Dat:
df = pd.read_csv('PhysioTrainerBot/data/strength_workouts.csv')
general_message = '''The most important insight/recommendation I can give you is that taking care of your physical health requires consistency.
Therefore, do not push your limits too far and try to get the most amount pleasure in your exercises.
If you're having a rough day and the idea of working out is dreadful, then just do the bare minimum  to give you the satisfaction of showing up.'''
stamina_message = '''Jogging is one of the most '''
# split data into upper and lower body
df_lower = pd.DataFrame()
df_other = pd.DataFrame()
for ind, entry in df.iterrows():
    if (1 == entry['Calves']) or (2 == entry['Calves']):
        df_lower = df_lower.append(entry)
    elif (1 == entry['Quadriceps']) or (2 == entry['Quadriceps']):
        df_lower = df_lower.append(entry)
    elif (1 == entry['Hamstrings']) or (2 == entry['Hamstrings']):
        df_lower = df_lower.append(entry)
    elif (1 == entry['Gluteus']) or (2 == entry['Gluteus']):
        df_lower = df_lower.append(entry)
    elif (1 == entry['Hipsother']) or (2 == entry['Hipsother']):
        df_lower = df_lower.append(entry)
    else: 
        df_other = df_other.append(entry)

# Create Class
class NonInjured():
    def __init__(self, goal):
        self.goal = goal
        print('') # print general reccomendation for physical health
    
    def get_plan(self):

        print('plan')

user1 = NonInjured('lower')