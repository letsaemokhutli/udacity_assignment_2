## import libraries
import sys
from Process_Utilities import load_data,clean_data,save_data
import warnings
warnings.filterwarnings('ignore')
import datetime
import pandas as pd
import numpy as np
import sqlite3
import os
import time

def main():
    if len(sys.argv) == 4:
        start_time = time.time()
        print('------------------------------------------------')
        current_directory = os.getcwd()
        print('Directory running from :',current_directory)
        print('------------------------------------------------')
        print("start time :",datetime.datetime.now())
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Extracting data...')
        print('------------------------------------------------')
        df = load_data(messages_filepath, categories_filepath)
        print('Transforming data...')
        print('------------------------------------------------')
        df = clean_data(df)
        print('Loading data...')
        print('------------------------------------------------')
        save_data(df, database_filepath)
        print('------------------------------------------------')
        end_time = time.time()
        print("end time",datetime.datetime.now())
        run_time = end_time - start_time
        print("pipe-line ran for",np.round(run_time,2),"seconds")
        print('------------------------------------------------')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()