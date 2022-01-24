# Disaster Response Pipeline Project

## Table of Contents
- Introduction & Project Motivation
- Installation
- Description
- Clone Repository
- Instructions

###Introduction & Project Motivation
The goal of the project is to classify the disaster messages into categories. In this project, I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages. Through a web app, the user can input a new message and get classification results in several categories. The web app also display visualizations of the data.


### Installation
The used Python libraries :


- numpy 
- pandas as pd
- sqlite3
- sqlalchemy 
- create_enginepandas
- os
- re
- nltk
- sklearn.pipeline 
- sklearn.base 
- sklearn.feature_extraction.text 
- sklearn.multioutput 
- sklearn.ensemble 
- sklearn.model_selection 
- sklearn.metrics 
- sklearn.model_selection 
- sklearn.metrics 
- pickle


### Description
1. ETL Pipeline: process_data.py file contain the script to create ETL pipline which:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database
2. ML Pipeline: train_classifier.py file contain the script to create ML pipline which:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes models
- Outputs results on the test set
- Exports the final model as a pickle file
3. Flask Web App: the web app enables the user to enter a disaster message in order to get the categorisation of the message.


### Clone Repository
You can clone this github repository : https://github.com/ordepzero/disaster_respnse_pipeline   


### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

