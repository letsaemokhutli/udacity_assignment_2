## import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
import pickle
import sqlite3
import warnings
warnings.filterwarnings('ignore')

def load_data(database_filepath):  
    """
    Load data from a SQLite database into a Pandas DataFrame.

    Parameters:
    -----------
    database_filepath : str
        Filepath to the SQLite database file.

    Returns:
    --------
    pandas.DataFrame or None
        Returns a DataFrame containing the data from the 'Disaster_table' table 
        in the SQLite database. If an error occurs during data loading, returns None.
    """
    conn = sqlite3.connect(database_filepath)
    query = f"""SELECT * FROM Disaster_table"""
    df = pd.read_sql_query(query, conn).head(5000)
    conn.close()
    return df

def build_model_XG_BOOST():
    """
    Build a pipeline for multi-output classification using XGBoost.

    Returns:
    --------
    sklearn.pipeline.Pipeline
        Returns a pipeline object configured with CountVectorizer for text 
        feature extraction and XGBoost classifier for multi-output classification.
    """
    pipeline_v2 = Pipeline([
    ('count_vectorizer', CountVectorizer()),  
    ('clf', MultiOutputClassifier(XGBClassifier()))  
    ])
    return pipeline_v2

def build_model_RF():
    """
    Build a pipeline for multi-output classification using Random Forest.

    Returns:
    --------
    sklearn.pipeline.Pipeline
        Returns a pipeline object configured with TfidfVectorizer for text 
        feature extraction and RandomForestClassifier for multi-output classification.
    """
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  
    ])
    return pipeline

#training random forests ml pipeline
def train_rf_pipeline(rf_ml_pipeline,X_train, X_test, y_train, y_test):
        """
        Train a random forest with TF-IDF text processing pipeline and evaluate its performance.

        Parameters:
        -----------
        rf_ml_pipeline : sklearn.pipeline.Pipeline
            Random forest with TF-IDF text processing pipeline.

        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.

        X_test : array-like or sparse matrix, shape (n_samples, n_features)
            Testing data.

        y_train : array-like, shape (n_samples, n_outputs)
            Multi-output target values for training.

        y_test : array-like, shape (n_samples, n_outputs)
            Multi-output target values for testing.

        Returns:
        --------
        None
            This function prints the performance report of the trained pipeline.
        """
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

def train_xgb_pipeline(xgb_ml_pipeline,X_train, X_test, y_train, y_test):
        """
        Train an XGBoost with count vectorizer text processing pipeline and evaluate its performance.

        Parameters:
        -----------
        xgb_ml_pipeline : sklearn.pipeline.Pipeline
            XGBoost with count vectorizer text processing pipeline.

        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.

        X_test : array-like or sparse matrix, shape (n_samples, n_features)
            Testing data.

        y_train : array-like, shape (n_samples, n_outputs)
            Multi-output target values for training.

        y_test : array-like, shape (n_samples, n_outputs)
            Multi-output target values for testing.

        Returns:
        --------
        None
            This function prints the performance report of the trained pipeline.
        """
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

def xgb_pipeline_tuned(xgb_ml_pipeline,X_train, y_train):
    """
    Tune hyperparameters of an XGBoost with count vectorizer text processing pipeline.

    Parameters:
    -----------
    xgb_ml_pipeline : sklearn.pipeline.Pipeline
        XGBoost with count vectorizer text processing pipeline.

    X_train : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.

    y_train : array-like, shape (n_samples, n_outputs)
        Multi-output target values for training.

    Returns:
    --------
    sklearn.pipeline.Pipeline
        Returns a tuned pipeline object with optimized hyperparameters.
    """
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

def rf_pipeline_tuned(rf_ml_pipeline,X_train, y_train):
    """
    Tune hyperparameters of a random forest with TF-IDF text processing pipeline.

    Parameters:
    -----------
    rf_ml_pipeline : sklearn.pipeline.Pipeline
        Random forest with TF-IDF text processing pipeline.

    X_train : array-like or sparse matrix, shape (n_samples, n_features)
        Training data.

    y_train : array-like, shape (n_samples, n_outputs)
        Multi-output target values for training.

    Returns:
    --------
    sklearn.pipeline.Pipeline
        Returns a tuned pipeline object with optimized hyperparameters.
    """
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

def save_model(model, model_filepath) :
    """
    Save a trained model to a file using pickle.

    Parameters:
    -----------
    model : object
        Trained model object to be saved.

    model_filepath : str
        Filepath where the model will be saved.

    Returns:
    --------
    None
        This function saves the model to the specified filepath.
    """
    print('-----------------------------------------------------------------------------')
    print("Exporting model artifacts:")
    print(model_filepath)
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)