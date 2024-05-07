## import libraries
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

# Loads data from db from ETL pipeline to dataframe for cleaning
def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    query = f"""SELECT * FROM Disaster_table"""
    df = pd.read_sql_query(query, conn).head(5000)
    conn.close()
    return df

#Iniitating ml pipeline using xgboost
def build_model_XG_BOOST():
    pipeline_v2 = Pipeline([
    ('count_vectorizer', CountVectorizer()),  
    ('clf', MultiOutputClassifier(XGBClassifier()))  
    ])
    return pipeline_v2
#Iniitating ml pipeline using random forests
def build_model_RF():
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  
    ])
    return pipeline

#training random forests ml pipeline
def train_rf_pipeline(rf_ml_pipeline,X_train, X_test, y_train, y_test):
        print("Initializing and training random forest, tifidf text porcessor pipeline...")
        print('----------------------------------------------------------------------------------')
        rf_ml_pipeline.fit(X_train, y_train)
        y_pred = rf_ml_pipeline.predict(X_test)
        y_test_flat = y_test.values.ravel()
        y_pred_flat = y_pred.ravel()
        report = classification_report(y_test_flat, y_pred_flat)
        print("Random forests,tifidf text porcessor Pipeline perfomance...")
        print('----------------------------------------------------------------------------------')
        print(report)

#training xgboost ml pipeline
def train_xgb_pipeline(xgb_ml_pipeline,X_train, X_test, y_train, y_test):
        print("Initializing and training xg boost,count vectorizer text porcessor pipeline...")
        print('----------------------------------------------------------------------------------')
        min_label = y_train.min()
        y_train_adjusted = y_train - min_label
        xgb_ml_pipeline.fit(X_train, y_train_adjusted)
        y_pred = xgb_ml_pipeline.predict(X_test)
        y_test_flat = y_test.values.ravel()
        y_pred_flat = y_pred.ravel()
        report = classification_report(y_test_flat, y_pred_flat)
        print("xg boost, count vectorizer text porcessor Pipeline perfomance...")
        print('----------------------------------------------------------------------------------')
        print(report)

#training tuned xgboost ml pipeline
def xgb_pipeline_tuned(xgb_ml_pipeline,X_train, y_train):
    print('(XGBClassifier,CountVectorizer) - Tuning model...')
    print('----------------------------------------------------------------------------------')
    min_label = y_train.min()
    y_train_adjusted = y_train - min_label
    param_grid = {
            'clf__estimator__n_estimators': [50, 100, 200],
            'clf__estimator__max_depth': [3, 5, 7],
            'count_vectorizer__max_features': [1000, 5000, 10000],  # Max features for CountVectorizer
            'count_vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],  # N-gram range for CountVectorizer
    }
    grid_search = GridSearchCV(estimator=xgb_ml_pipeline, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train_adjusted)
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)
    pipeline_v2 = Pipeline([
            ('count_vectorizer', CountVectorizer()),  
            ('clf', MultiOutputClassifier(XGBClassifier(clf__estimator__max_depth= 3, clf__estimator__n_estimators= 50, count_vectorizer__max_features= 1000, count_vectorizer__ngram_range= (1, 1))))  # Multi-output classifier
    ])
    return pipeline_v2

#training tuned random forest ml pipeline
def rf_pipeline_tuned(rf_ml_pipeline,X_train, y_train):
    print('(Random Forests,tifidf) - Tuning model...')
    print('----------------------------------------------------------------------------------')
    param_grid = {
            'tfidf__max_features': [1000, 2000, 3000],  # Number of features to consider
            'tfidf__ngram_range': [(1, 1), (1, 2)],      # Range of n-grams
            'clf__estimator__n_estimators': [100, 200, 300],  # Number of trees in the forest
            'clf__estimator__max_depth': [10, 20, 30] 
    }
    grid_search = GridSearchCV(rf_ml_pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    print("Best Parameters:",best_pipeline) 
    print('-----------------------------------------------------------------------------')
    # Define pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))), 
        ('clf', MultiOutputClassifier(RandomForestClassifier(
            max_depth=30,n_estimators=200
        )))  
    ])
    return pipeline

#Exporting model artifact as pickle file
def save_model(model, model_filepath) :
    print('-----------------------------------------------------------------------------')
    print("Exporting model artifacts:")
    print(model_filepath)
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)