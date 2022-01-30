import sys
import pandas as pd 
import os
from sqlalchemy import create_engine
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    Load the data from Database and return in the requested format.
    
    Input:
    database_filepath(path/file): for the input database
    
    Output:
    X, Y, catagory_names as requested """
    
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql("DisasterResponse", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = y.columns
    
    return X, Y, category_names

def tokenize(text):
    """
    Tokenise the input message.
    
    Input:
    text(str): message
    
    Output:
    lemm(list of str): a list compose of the message words """
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words = text.split()
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmed
    


def build_model():
     """
    Build model with parameter optimization.
    
    Input: -
    
    Output: Model
    
    Note : Can take time """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'vect__max_df': (0.5, 1.0),
    'clf__estimator__n_estimators': (50, 100, 200),
    'tfidf__use_idf': (True, False)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def perf_by_classification_report(y_test, y_pred):
    """
    Test model with sklearn's classification_report
    
    Inputs: 
    y_test, split data
    y_pred, split data
    
    Output:
    Prints results """
    
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))
    average_accuracy = (y_pred == y_test.values).mean()
    print('Average_accuracy {:.3f}'.format(average_accuracy))


def evaluate_model(model, X_test, Y_test, category_names):
     """
    Test the model with the function perf_by_classification_report.
    
    Input:
    X, Y, catagory_names as requested
    
    Output: The results of the evaluation
     """
        
    y_pred = model.predict(X_test)
    perf_by_classification_report(Y_test, y_pred)
    


def save_model(model, model_filepath):
    """ Save the model by using pickle"""
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()