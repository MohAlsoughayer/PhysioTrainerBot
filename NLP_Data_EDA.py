# Author: Mohammed Alsoughayer
# Descritpion: Perform EDA on the gathered data to narrow down the NLP Datasets for the chatbot 
# %% 
# Import necessary packages
import numpy as np
import pandas as pd
import math
from plotnine import ggplot, aes, geom_density, geom_line, geom_point, ggtitle, labs
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Modeling process
# import xgboost as xgb
# from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn import linear_model
# from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.metrics import mean_squared_error, roc_auc_score
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import make_column_selector as selector
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# %%
