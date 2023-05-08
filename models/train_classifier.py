import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
import joblib
import time
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
# from ..transformers import NounProportion, CapitalWordsCount, WordsCount
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
# from ..utils.tokenize import tokenize
from utils.tokenize import tokenize
from transformers.CapitalWordsCount import CapitalWordsCount
from transformers.NounProportion import NounProportion
from transformers.WordsCount import WordsCount
from collections import defaultdict

def load_data(database_filepath):
    '''Reads data from the database and loads it into dataframes
    
    Args:
    database_filepath (string) - path to the database file

    Returns:
    X - dataframe containing 'message' column;
    Y - dataframe containing all categories columns;
    category_names - list with all the categories names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesCategories', con=engine.connect())

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns

    return X, Y, category_names

def build_model():
    '''Builds the model using a Pipeline consisting of TfidfVectorizer, MultiOutputClassifier, RandomForestClassifier
    
    Args:
    None

    Returns:
    cv - the model obtained by performing a GridSearchCV over the pipeline
    '''
    n_cpu = os.cpu_count()
    print("Number of CPUs in the system: %s. Will use %s." % (n_cpu, (n_cpu - 1)))

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('trans', TfidfVectorizer(tokenizer=tokenize)),
            ('nounProportion', NounProportion()),
            ('wordsCount', WordsCount()),
            ('capitalWordsCount', CapitalWordsCount())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=n_cpu - 1)))
    ])

    # parameters = {
    #     'clf__estimator__n_estimators': [50, 100, 200, 300],
    #     'clf__estimator__min_samples_split': [2, 5, 10],
    #     # 'clf__estimator__min_samples_leaf': [1, 2, 4],
    #     'clf__estimator__max_depth': [10, 50, None],
    #     'clf__estimator__bootstrap': [True, False],
    #     # 'clf__estimator__max_features': ['sqrt', 'log2', None],
    # }
    parameters = {}

    print("Parameters: %s" % parameters)

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names, test_name):
    '''Evaluates a model by displaying the classification reports for all the categories
    
    Args:
    model - the model to evaluate
    X_test - the input dataframe to test
    Y_test - the output dataframe to test
    category_names - list containing the names of the categories

    Returns:
    None
    '''
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)

    macro_avg_results = defaultdict(dict)
    weighted_avg_results = defaultdict(dict)
    for column in category_names:
        report_dict = classification_report(Y_test[column], Y_pred_df[column], output_dict=True)
        report = classification_report(Y_test[column], Y_pred_df[column])

        report_dict['weighted avg'].pop('support', None)
        report_dict['macro avg'].pop('support', None)
        weighted_avg_results[column] = report_dict['weighted avg']
        macro_avg_results[column] = report_dict['macro avg']

        print('Report for ' + column + ' category')
        print(report)
    
    if (test_name == None):
        test_name = 'testUnnamed'

    macro_avg_results_df = pd.DataFrame.from_dict(macro_avg_results, orient='index')
    weighted_avg_results_df = pd.DataFrame.from_dict(weighted_avg_results, orient='index')

    with open('tests/' + test_name + '_weightedAvg.md', 'w') as fid:
        print(weighted_avg_results_df.to_markdown(), file=fid)

    with open('tests/' + test_name + '_macroAvg.md', 'w') as fid:
        print(macro_avg_results_df.to_markdown(), file=fid)


def save_model(model, model_filepath):
    joblib.dump(model, open(model_filepath, 'wb'))


# def tokenize(text):
#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()

#     text = text.lower()
#     text = re.sub("[^a-zA-Z0-9]", " ", text)

#     words_list = word_tokenize(text)
#     stopwords_list = stopwords.words('english')

#     words_list = [word for word in words_list if word not in stopwords_list]

#     words_list = [lemmatizer.lemmatize(word) for word in words_list]
#     words_list = [lemmatizer.lemmatize(word, pos='v') for word in words_list]

#     words_list = [stemmer.stem(word) for word in words_list]

#     return words_list


def main():
    if len(sys.argv) >= 3:
        start_time = time.time()

        database_filepath, model_filepath, test_description, test_name = sys.argv[1:]
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

        print("\n The best parameters across ALL searched params:\n", model.best_params_)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, test_name)

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