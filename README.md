# Disaster response app

## Background 

### This is an assignment intended for testing software development and data engineering skills I acquired from udacity course I am taking

###### The task was to build a web app that uses Machine learning to classifiy if text input by the user is disaster related or not. 
###### Data Engineeing came into place when an ETL pipeline was built to populate training data for the Multioutput classifer used for classification. 
###### Software engineering came into play when a web app was used to build a Python Flask app.
###### In completion of the project, it's benefit will be that it can be used by organisations in disaster response

## How to run the app
###### 1. On your terminal run python process_data.py messages.csv categories.csv DisasterResponse.db to run the ETL pipeline
###### 2. On your terminal run python train_classifier.py DisasterResponse.db final_model.pkl to run the ML pipeline
###### 3. On your terminal run python run.py or use an IDE to run the web app
###### 4. File structure

###### udacity_assigment_2
######      |___.git/
######       |___categories.csv
######       |___Classifier_utility.py
######       |___DisasterResponse.db
######       |___final_model.pkl
######       |___genre_count.csv
######       |___interface.js
######       |___interface.py
      |___messages.csv
      |___process_data.py
      |___Process_Utilities.py
      |___Procfile
      |___README.md
      |___requirements.txt
      |___run.py

      |___templates/
      |___train_classifier.py
      |___venv/
      |_____pycache__/

### Dependencies are on the requirements.txt
### https://medium.com/@mokhutliletsae/01d5334b8493, project article
