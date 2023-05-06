import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import time
import os
from custom_transformers.NounProportion import NounProportion
from custom_transformers.WordsCount import WordsCount
from custom_transformers.CapitalWordsCount import CapitalWordsCount
from utils import tokenize


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', con=engine.connect())

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns

    return X, Y, category_names

def build_model():
    n_cpu = os.cpu_count()
    print("Number of CPUs in the system: %s. Will use %s." % (n_cpu, (n_cpu - 1)))

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('trans', TfidfVectorizer(tokenizer=tokenize)),
            # ('nounProportion', NounProportion()),
            # ('wordsCount', WordsCount())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=n_cpu - 1)))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 5, 10],
        # 'clf__estimator__min_samples_leaf': [1, 2, 4],
        'clf__estimator__max_depth': [10, 30, 70, None],
        # 'clf__estimator__bootstrap': [True, False]
    }

    print("Parameters: %s" % parameters)

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

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
    if len(sys.argv) >= 3:
        start_time = time.time()


        # nounProportion = CapitalWordsCount()
        # print('-----')
        # print(nounProportion.transform(['I am in Romania and starving, at the University we need food and water']))

        database_filepath, model_filepath, test_description = sys.argv[1:]
        print('Test description:\n')
        print(test_description)
        print ('\n\n')

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
        
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        end_time = time.time()

        print('Execution time: %s minutes' % ((end_time - start_time) / 60))
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()