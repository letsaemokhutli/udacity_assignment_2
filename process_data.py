import sys
# import libraries
import warnings
warnings.filterwarnings('ignore')
import datetime
import pandas as pd
import numpy as np
import sqlite3
import os
import time

def load_data(messages_filepath, categories_filepath):
    try :
        # load datasets
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        messages_genre = messages.groupby(['genre']).size().reset_index(name='count')
        messages_genre.to_csv("genre_count.csv",index=False)
        # merge datasets
        merged_df = pd.merge(messages, categories, on='id', how='left')
        # create a dataframe of the 36 individual category columns
        categories_split = categories['categories'].str.split(';',expand=True)
        categories_split = categories_split.add_prefix('category_')
        categories = pd.concat([categories, categories_split], axis=1)
        # select the first row of the categories dataframe
        row = categories.head(1)
        # use this row to extract a list of new column names for categories.
        # one way is to apply a lambda function that takes everything 
        # up to the second to last character of each string with slicing
        category_colnames = row.columns
        categories.columns = category_colnames
        categories[categories.drop(["categories","id"],axis=1).columns] = categories[categories.drop(["categories","id"],axis=1).columns].apply(lambda x: x.str[-1])
        categories[categories.drop(["categories","id"],axis=1).columns] = categories[categories.drop(["categories","id"],axis=1).columns].apply(pd.to_numeric)
        merged_df = categories.copy()
        # concatenate the original dataframe with the new `categories` dataframe
        merged_df = merged_df.merge(messages, on='id', how='left')
        #using one hot encoding for nominal columns
        merged_df = pd.get_dummies(merged_df.drop(["original"],axis=1), columns=["genre"]) 
    except ValueError as error:
        print(error)
    return merged_df

def clean_data(df):
    try :
        # check number of duplicates
        duplicate_rows = df[df.duplicated()]
        # Counting duplicates
        num_duplicates = len(duplicate_rows)
        # drop duplicates
        df = df.drop_duplicates()
        # check number of duplicates
        duplicate_rows = df[df.duplicated()]
        # Counting duplicates, confirming if all dutes were removed
        num_duplicates = len(duplicate_rows)
    except ValueError as error:
        print(error)
    return df

def save_data(df, database_filename):
    conn = sqlite3.connect(database_filename)
    #engine = create_engine('sqlite:///InsertDatabaseName.db')
    #df.to_sql('InsertTableName', engine, index=False)
    df.to_sql(database_filename, conn, if_exists='replace', index=False)
    # Commit changes and close the connection
    conn.commit()
    conn.close() 

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