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
import numpy as np
import os
import datetime
import time
import warnings
warnings.filterwarnings('ignore')
import sys


def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    # Read data from SQLite database into a DataFrame
    query = "SELECT * FROM etl_disaster_table"
    df = pd.read_sql_query(query, conn).head(5000)
    # Close the connection
    conn.close()
    return df

def build_model():
    pass


def evaluate_model(model, X_test, Y_test):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        training_data = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(
            training_data['message'], training_data.drop(["message","id","categories"],axis=1), test_size=0.2, random_state=42
        )
        #print('Building model...')
        #model = build_model()
        
        #print('Training model...')
        #model.fit(X_train, Y_train)
        
        #print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)

        #print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #save_model(model, model_filepath)

        #print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()