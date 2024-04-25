# import libraries
import warnings
warnings.filterwarnings('ignore')
import datetime
import pandas as pd
import numpy as np
import sqlite3
import os
import time
# Record the start time
start_time = time.time()
print('------------------------------------------------')
current_directory = os.getcwd()
print('Directory running from :',current_directory)
print('------------------------------------------------')
print("start time :",datetime.datetime.now())
#--------------------------------------------Extract-----------------------------------------------------
print('------------------------------------------------')
print("Perfoming Exctract process...")
try :
    # load datasets
    messages_file = f"""{current_directory}\\messages.csv"""
    categories_file = f"""{current_directory}\\categories.csv"""
    messages = pd.read_csv(messages_file)
    categories = pd.read_csv(categories_file)
    print("Exctraction completed!")
except ValueError as error:
    print(error)
#--------------------------------------------Transform-----------------------------------------------------
print('------------------------------------------------')
print("Perfoming Transform process...")
try :
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
    # check number of duplicates
    duplicate_rows = merged_df[merged_df.duplicated()]
    # Counting duplicates
    num_duplicates = len(duplicate_rows)
    # drop duplicates
    merged_df = merged_df.drop_duplicates()
    # check number of duplicates
    duplicate_rows = merged_df[merged_df.duplicated()]
    # Counting duplicates, confirming if all duplicates were removed
    num_duplicates = len(duplicate_rows)
    print("Transformation completed!")
except ValueError as error:
    print(error)
#--------------------------------------------Load-----------------------------------------------------
try :
    print('------------------------------------------------')
    print("Perfoming Load process...")
    #creating sqllite connection
    conn = sqlite3.connect('etl_disaster_data.db')
    #writing the merged df to the db
    merged_df.to_sql('etl_disaster_table', conn, if_exists='replace', index=False)
    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print("Load completed!")
    print('------------------------------------------------')
    end_time = time.time()
    print("end time",datetime.datetime.now())
    run_time = end_time - start_time
    print("pipe-line ran for",np.round(run_time,2),"seconds")
    print('------------------------------------------------')
except ValueError as error:
    print(error)