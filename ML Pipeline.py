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
print('--------------------------------------------------------')
current_directory = os.getcwd()
print('Directory running from :',current_directory)
print('--------------------------------------------------------')
print("start time :",datetime.datetime.now())
# load data from database
#engine = create_engine('sqlite:///InsertDatabaseName.db')
print('--------------------------------------------------------')
print('Reading data from the db into training dataframe...')
conn = sqlite3.connect('etl_disaster_data.db')
# Read data from SQLite database into a DataFrame
query = "SELECT * FROM etl_disaster_table"
df = pd.read_sql_query(query, conn).head(5000)
# Close the connection
conn.close()
print('--------------------------------------------------------')
print('Preparing training and test data...')
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df.drop(["message","id"],axis=1), test_size=0.2, random_state=42
)
print('--------------------------------------------------------')
print('Initializing ML pipeline...')
# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Text preprocessing
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classifier
])
print('--------------------------------------------------------')
print('Fitting ML pipeline...')
### 4. Train pipeline
pipeline.fit(X_train, y_train)