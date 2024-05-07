# import libraries
from Classifier_utility import load_data,build_model_RF,build_model_XG_BOOST,train_rf_pipeline,train_xgb_pipeline,rf_pipeline_tuned,xgb_pipeline_tuned,save_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
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

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('----------------------------------------------------------------------------------')
        print('Loading data...\nDATABASE: {}'.format(database_filepath))
        training_data = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(
            training_data['message'], training_data.drop(["message","id","categories"],axis=1), test_size=0.2, random_state=42
        )
        start_time = time.time()
        print('----------------------------------------------------------------------------------')
        current_directory = os.getcwd()
        print('Directory running from :',current_directory)
        print('----------------------------------------------------------------------------------')
        print("start time :",datetime.datetime.now())
        print('----------------------------------------------------------------------------------')
        rf_ml_pipeline = build_model_RF()
        xgb_ml_pipeline = build_model_XG_BOOST()
        train_rf_pipeline(rf_ml_pipeline,X_train, X_test, y_train, y_test)
        print('----------------------------------------------------------------------------------')
        train_xgb_pipeline(xgb_ml_pipeline,X_train, X_test, y_train, y_test)
        print('----------------------------------------------------------------------------------')
        optimized_rf_pipeline = rf_pipeline_tuned(rf_ml_pipeline,X_train,y_train)
        optimized_xgb_pipeline = xgb_pipeline_tuned(xgb_ml_pipeline,X_train,y_train)
        train_rf_pipeline(optimized_rf_pipeline,X_train, X_test, y_train, y_test)
        train_xgb_pipeline(optimized_xgb_pipeline,X_train, X_test, y_train, y_test)
        print("Load completed!")
        print('------------------------------------------------')
        end_time = time.time()
        print("end time",datetime.datetime.now())
        run_time = end_time - start_time
        print("pipe-line ran for",np.round(run_time/60,2),"minutes")
        print('------------------------------------------------')
        save_model(xgb_ml_pipeline,model_filepath)
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()