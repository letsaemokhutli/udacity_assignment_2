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
# Record the start time
start_time = time.time()
print('-----------------------------------------------------------------------------')
current_directory = os.getcwd()
print('Directory running from :',current_directory)
print('-----------------------------------------------------------------------------')
print("start time :",datetime.datetime.now())
# load data from database
#engine = create_engine('sqlite:///InsertDatabaseName.db')
print('-----------------------------------------------------------------------------')
print('Reading data from the db into training dataframe...')
conn = sqlite3.connect('etl_disaster_data.db')
# Read data from SQLite database into a DataFrame
query = "SELECT * FROM etl_disaster_table"
df = pd.read_sql_query(query, conn).head(5000)
# Close the connection
conn.close()
print('-----------------------------------------------------------------------------')
print('Preparing training and test data...')
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df.drop(["message","id","categories"],axis=1), test_size=0.2, random_state=42
)
print('-----------------------------------------------------------------------------')
print('Initializing ML pipeline...')
# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Text preprocessing
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Multi-output classifier
])
print('-----------------------------------------------------------------------------')
print('Fitting ML pipeline...(RandomForestClassifier,TfidfVectorizer)')
### 4. Train pipeline
pipeline.fit(X_train, y_train)
print('-----------------------------------------------------------------------------')
print('Computing model score...')
y_pred = pipeline.predict(X_test)
# Calculate accuracy for each label
accuracies = [accuracy_score(y_test[label], y_pred[:, idx]) for idx, label in enumerate(y_test.columns)]
#print("Accuracy for each label:", accuracies)
# Calculate Hamming Loss for each label
hamming_losses = [hamming_loss(y_test[label], y_pred[:, idx]) for idx, label in enumerate(y_test.columns)]
#print("Hamming Loss for each label:", hamming_losses)
# Calculate Jaccard Score for each label
jaccard_scores = [jaccard_score(y_test[label], y_pred[:, idx], average=None) for idx, label in enumerate(y_test.columns)]
#print("Jaccard Score for each label:", jaccard_scores)
# Calculate F1 Score for each label
f1_scores = [f1_score(y_test[label], y_pred[:, idx], average=None) for idx, label in enumerate(y_test.columns)]
#print("F1 Score for each label:", f1_scores)
print('-----------------------------------------------------------------------------')
print("Classification Report:")
# Assuming y_test and y_pred are your true labels and predicted labels respectively
# Flatten y_test and y_pred to fit classification_report
y_test_flat = y_test.values.ravel()
y_pred_flat = y_pred.ravel()
# Generate the classification report
report = classification_report(y_test_flat, y_pred_flat)
print(report)
print('-----------------------------------------------------------------------------')
print("Hyper-parameter tuning...")
"""param_grid = {
    'tfidf__max_features': [1000, 2000, 3000],  # Number of features to consider
    'tfidf__ngram_range': [(1, 1), (1, 2)],      # Range of n-grams
    'clf__estimator__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'clf__estimator__max_depth': [10, 20, 30],       # Maximum depth of the tree
}

# Perform grid search cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)
# Fit grid search on training data
grid_search.fit(X_train, y_train)

# Evaluate performance on test set
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)
print("Optimal parameters :",best_pipeline) """
print('-----------------------------------------------------------------------------')
print("Retraining the model with tuned parameters...")
# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),  # Text preprocessing
    ('clf', MultiOutputClassifier(RandomForestClassifier(
        max_depth=30,n_estimators=200
    )))  # Multi-output classifier
])
#Train pipeline
pipeline.fit(X_train, y_train)
print('-----------------------------------------------------------------------------')
print('recomputing model score...')
y_pred = pipeline.predict(X_test)
# Calculate accuracy for each label
accuracies = [accuracy_score(y_test[label], y_pred[:, idx]) for idx, label in enumerate(y_test.columns)]
#print("Accuracy for each label:", accuracies)

# Calculate Hamming Loss for each label
hamming_losses = [hamming_loss(y_test[label], y_pred[:, idx]) for idx, label in enumerate(y_test.columns)]
#print("Hamming Loss for each label:", hamming_losses)

# Calculate Jaccard Score for each label
jaccard_scores = [jaccard_score(y_test[label], y_pred[:, idx], average=None) for idx, label in enumerate(y_test.columns)]
#print("Jaccard Score for each label:", jaccard_scores)

# Calculate F1 Score for each label
f1_scores = [f1_score(y_test[label], y_pred[:, idx], average=None) for idx, label in enumerate(y_test.columns)]
#print("F1 Score for each label:", f1_scores)
print('-----------------------------------------------------------------------------')
print("Classification Report(For model with optimal parameters):")
# Assuming y_test and y_pred are your true labels and predicted labels respectively
# Flatten y_test and y_pred to fit classification_report
y_test_flat = y_test.values.ravel()
y_pred_flat = y_pred.ravel()
# Generate the classification report
report = classification_report(y_test_flat, y_pred_flat)
print("Classification Report:")
print(report)
print('-----------------------------------------------------------------------------')
print("Trying a different model with a different text processor - (XGBClassifier,CountVectorizer):")
# Define pipeline with CountVectorizer and use XGboost classifier
pipeline_v2 = Pipeline([
    ('count_vectorizer', CountVectorizer()),  # Text preprocessing
    ('clf', MultiOutputClassifier(XGBClassifier()))  # Multi-output classifier
])
# Find the minimum value in y_train
min_label = y_train.min()
# Adjust the target labels
y_train_adjusted = y_train - min_label
# Fit the pipeline with the adjusted labels
pipeline_v2.fit(X_train, y_train_adjusted)
print('-----------------------------------------------------------------------------')
print("Classification Report(For model with optimal parameters) and scores:")
y_pred = pipeline_v2.predict(X_test)
# Calculate accuracy for each label
accuracies = [accuracy_score(y_test[label], y_pred[:, idx]) for idx, label in enumerate(y_test.columns)]
#print("Accuracy for each label:", accuracies)

# Calculate Hamming Loss for each label
hamming_losses = [hamming_loss(y_test[label], y_pred[:, idx]) for idx, label in enumerate(y_test.columns)]
#print("Hamming Loss for each label:", hamming_losses)

# Calculate Jaccard Score for each label
jaccard_scores = [jaccard_score(y_test[label], y_pred[:, idx], average=None) for idx, label in enumerate(y_test.columns)]
#print("Jaccard Score for each label:", jaccard_scores)

# Calculate F1 Score for each label
f1_scores = [f1_score(y_test[label], y_pred[:, idx], average=None) for idx, label in enumerate(y_test.columns)]
#print("F1 Score for each label:", f1_scores)
# Assuming y_test and y_pred are your true labels and predicted labels respectively

# Flatten y_test and y_pred to fit classification_report
y_test_flat = y_test.values.ravel()
y_pred_flat = y_pred.ravel()

# Generate the classification report
report = classification_report(y_test_flat, y_pred_flat)

print("Classification Report:")
print(report)
print('-----------------------------------------------------------------------------')
print("Parameter tuning for XGboost:")
"""pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer()),  # Text preprocessing
    ('clf', MultiOutputClassifier(XGBClassifier()))  # Multi-output classifier
])

# Find the minimum value in y_train
min_label = y_train.min()

# Adjust the target labels
y_train_adjusted = y_train - min_label
param_grid = {
    'clf__estimator__n_estimators': [50, 100, 200],
    'clf__estimator__max_depth': [3, 5, 7],
    'count_vectorizer__max_features': [1000, 5000, 10000],  # Max features for CountVectorizer
    'count_vectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],  # N-gram range for CountVectorizer
    # Add more hyperparameters as needed
}
# Initialize GridSearchCV with adjusted target labels
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='accuracy')

# Perform grid search with adjusted target labels
grid_search.fit(X_train, y_train_adjusted)

# Print best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)"""
print('-----------------------------------------------------------------------------')
print("Trying a optimized model - (XGBClassifier,CountVectorizer):")
# Define pipeline with CountVectorizer and use XGboost classifier
pipeline_v2 = Pipeline([
    ('count_vectorizer', CountVectorizer()),  # Text preprocessing
    ('clf', MultiOutputClassifier(XGBClassifier(clf__estimator__max_depth= 3, clf__estimator__n_estimators= 50, count_vectorizer__max_features= 1000, count_vectorizer__ngram_range= (1, 1))))  # Multi-output classifier
])
# Find the minimum value in y_train
min_label = y_train.min()
# Adjust the target labels
y_train_adjusted = y_train - min_label
# Fit the pipeline with the adjusted labels
pipeline_v2.fit(X_train, y_train_adjusted)
print('-----------------------------------------------------------------------------')
print("Classification Report(For model with optimal parameters) and scores:")
y_pred = pipeline_v2.predict(X_test)
# Calculate accuracy for each label
accuracies = [accuracy_score(y_test[label], y_pred[:, idx]) for idx, label in enumerate(y_test.columns)]
#print("Accuracy for each label:", accuracies)

# Calculate Hamming Loss for each label
hamming_losses = [hamming_loss(y_test[label], y_pred[:, idx]) for idx, label in enumerate(y_test.columns)]
#print("Hamming Loss for each label:", hamming_losses)

# Calculate Jaccard Score for each label
jaccard_scores = [jaccard_score(y_test[label], y_pred[:, idx], average=None) for idx, label in enumerate(y_test.columns)]
#print("Jaccard Score for each label:", jaccard_scores)

# Calculate F1 Score for each label
f1_scores = [f1_score(y_test[label], y_pred[:, idx], average=None) for idx, label in enumerate(y_test.columns)]
#print("F1 Score for each label:", f1_scores)
# Assuming y_test and y_pred are your true labels and predicted labels respectively

# Flatten y_test and y_pred to fit classification_report
y_test_flat = y_test.values.ravel()
y_pred_flat = y_pred.ravel()

# Generate the classification report
report = classification_report(y_test_flat, y_pred_flat)

print("Classification Report:")
print(report)
print('-----------------------------------------------------------------------------')
print("Exporting model artifacts:")
# Serialize the pipeline using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(pipeline_v2, file)


print("Load completed!")
print('------------------------------------------------')
end_time = time.time()
print("end time",datetime.datetime.now())
run_time = end_time - start_time
print("pipe-line ran for",np.round(run_time/60,2),"minutes")
print('------------------------------------------------')
    
