import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', con=engine.connect())

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    text = text.lower()
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    
    words_list = word_tokenize(text)
    stopwords_list = stopwords.words('english')
    
    words_list = [word for word in words_list if word not in stopwords_list]
    
    words_list = [lemmatizer.lemmatize(word) for word in words_list]
    words_list = [lemmatizer.lemmatize(word, pos='v') for word in words_list]

    words_list = [stemmer.stem(word) for word in words_list]
    
    return words_list


def build_model():
    pipeline = Pipeline([
        ('trans', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__min_samples_leaf': [1, 2, 4],
        'clf__estimator__max_features': ['auto', 'sqrt'],
        'clf__estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'clf__estimator__bootstrap': [True, False]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)

    for column in category_names:
        print('Report for ' + column + ' category')
        print(classification_report(Y_test[column], Y_pred_df[column]))


def save_model(model, model_filepath):
    joblib.dump(model, open(model_filepath, 'wb'))


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