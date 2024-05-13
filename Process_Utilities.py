## import libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import sqlite3
import nltk
import re
nltk.download('punkt') 
from nltk.corpus import stopwords
import re

def tokenize_text(df) :
    """
    Tokenize text messages in a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing text messages in the column named 'message'.

    Returns:
    --------
    pandas.DataFrame
        Returns the DataFrame with tokenized text messages in the 'message' column.
    """
    stop_words = set(stopwords.words('english'))
    df['message'] = df['message'].astype(str)
    df['message'] = df['message'].astype(str).apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
    df['message'] = df['message'].apply(lambda x: ' '.join([word for word in nltk.word_tokenize(x) if word not in stop_words]))
    return df

def load_data(messages_filepath, categories_filepath):
    """
    Load and preprocess messages and categories data.

    Parameters:
    -----------
    messages_filepath : str
        Filepath to the CSV file containing messages data.

    categories_filepath : str
        Filepath to the CSV file containing categories data.

    Returns:
    --------
    pandas.DataFrame
        Returns a DataFrame containing the merged and preprocessed messages and categories data.
    """
    try :
        messages = pd.read_csv(messages_filepath)
        messages = tokenize_text(messages)
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
        numeric_columns = merged_df.drop("id",axis=1).select_dtypes(include='number').columns
        # Boolean indexing to select rows with only 0 or 1 values in numeric columns
        mask = (merged_df[numeric_columns] == 0) | (merged_df[numeric_columns] == 1)
        merged_df = merged_df[mask.all(axis=1)]

    except ValueError as error:
        print(error)
    return merged_df

def clean_data(df):
    """
    Clean the input DataFrame by filling NaN values with 0 and removing duplicate rows.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to be cleaned.

    Returns:
    --------
    pandas.DataFrame
        Returns the cleaned DataFrame.
    """
    try :
        df = df.fillna(0)
        df = df.drop_duplicates()
    except ValueError as error:
        print(error)
    return df

def save_data(df, database_filename):
    """
    Save DataFrame to a SQLite database.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to be saved to the database.

    database_filename : str
        Filename of the SQLite database.

    Returns:
    --------
    None
        This function saves the DataFrame to the specified SQLite database.
    """
    conn = sqlite3.connect(database_filename)
    df.to_sql(name='Disaster_table', con = conn, if_exists='replace', index=False)
    conn.commit()
    conn.close() 