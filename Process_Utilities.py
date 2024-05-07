# import libraries
import sys
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
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        messages_genre = messages.groupby(['genre']).size().reset_index(name='count')
        messages_genre.to_csv("genre_count.csv",index=False)
        merged_df = pd.merge(messages, categories, on='id', how='left')
        categories_split = categories['categories'].str.split(';',expand=True)
        categories_split = categories_split.add_prefix('category_')
        categories = pd.concat([categories, categories_split], axis=1)
        row = categories.head(1)
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
        df = df.fillna(0)
        df = df.drop_duplicates()
    except ValueError as error:
        print(error)
    return df

def save_data(df, database_filename):
    conn = sqlite3.connect(database_filename)
    df.to_sql(name='Disaster_table', con = conn, if_exists='replace', index=False)
    conn.commit()
    conn.close() 