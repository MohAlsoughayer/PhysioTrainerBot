# Author: Mohammed Alsoughayer
# Descritpion: Gather Data and save it in a csv file 

# Import necessary packages
import numpy as np
import pandas as pd
import math
import seaborn as sns

# Create Class 
class HealthProfile():
    def __init__(self, age):
        self.age = age
        self.symptoms = []
        self.location = []
        self.description = ''
