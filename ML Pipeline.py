# import libraries
from sklearn.metrics import hamming_loss, jaccard_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
import nltk
import pickle
nltk.download('punkt') 
import sqlite3
import os
import datetime
import time
import warnings
warnings.filterwarnings('ignore')
# Record the start time
start_time = time.time()
print('------------------------------------------------')
current_directory = os.getcwd()
print('Directory running from :',current_directory)
print('------------------------------------------------')
print("start time :",datetime.datetime.now())