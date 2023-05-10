from collections import defaultdict
from transformers.WordsCount import WordsCount
from transformers.NounProportion import NounProportion
from transformers.CapitalWordsCount import CapitalWordsCount
from utils.tokenize import tokenize
import sys
from sqlalchemy import create_engine
import pandas as pd
import joblib
import time
import os
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def load_data(database_filepath, file):
    '''Reads data from the database and loads it into dataframes

    Args:
    database_filepath (string) - path to the database file

    Returns:
    X - dataframe containing 'message' column;
    Y - dataframe containing all categories columns;
    category_names - list with all the categories names
    '''
    print_('Loading data from {} database'.format(database_filepath), file)

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('MessagesCategories', con=engine.connect())

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns

    return X, Y, category_names


def build_model(file):
    '''Builds the model using a Pipeline consisting of TfidfVectorizer, MultiOutputClassifier, RandomForestClassifier

    Args:
    None

    Returns:
    cv - the model obtained by performing a GridSearchCV over the pipeline
    '''
    n_cpu = os.cpu_count()

    print('\nBuilding the model...\n')
    print_('\nNumber of CPUs in the system: %s. Will use %s for the classifier.\n' %
              (n_cpu, (n_cpu - 1)), file)
    
    if file:
        print('\nNumber of CPUs in the system: %s. Will use %s for the classifier.\n' %
            (n_cpu, (n_cpu - 1)))

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('trans', TfidfVectorizer(tokenizer=tokenize)),
            ('nounProportion', NounProportion()),
            ('wordsCount', WordsCount()),
            ('capitalWordsCount', CapitalWordsCount())
        ])),
        ('clf', MultiOutputClassifier(SVC()))
        # ('clf', MultiOutputClassifier(KNeighborsClassifier(n_jobs=n_cpu - 1)))
        # ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=n_cpu - 1)))
    ])

    print(pipeline.get_params())

    # Parameters for RandomForestClassifier
    # parameters = {
    #     'clf__estimator__n_estimators': [100, 200],
    #     'clf__estimator__min_samples_split': [2, 5, 10],
    #     'clf__estimator__max_depth': [10, 50, None],
    #     'clf__estimator__bootstrap': [True, False],
    # }

    # Parameters for KNeighborsClassifier
    # parameters = {
    #     'clf__estimator__n_neighbors': [5, 10, 20, 30],
    #     'clf__estimator__leaf_size': [15, 30, 50],
    #     'clf__estimator__p': [1, 2]
    # }

    # Parameters for SVC
    parameters = {
        'clf__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'clf__estimator__gamma': ['scale', 'auto'],
        'clf__estimator__degree': [2, 3, 4]
    }


    print_("\nParameters: {}\n".format(parameters), file=file)

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names, test_name, file):
    '''Evaluates a model by displaying the classification reports for all the categories

    Args:
    model - the model to evaluate
    X_test - the input dataframe to test
    Y_test - the output dataframe to test
    category_names - list containing the names of the categories

    Returns:
    None
    '''
    print('\nEvaluating the model...\n', file)

    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)

    if test_name and file:
        macro_avg_results = defaultdict(dict)
        weighted_avg_results = defaultdict(dict)

        for column in category_names:
            report = classification_report(Y_test[column], Y_pred_df[column])
            report_dict = classification_report(
                Y_test[column], Y_pred_df[column], output_dict=True)

            report_dict['weighted avg'].pop('support', None)
            report_dict['macro avg'].pop('support', None)
            weighted_avg_results[column] = report_dict['weighted avg']
            macro_avg_results[column] = report_dict['macro avg']

            print_('Report for {} category'.format(column), file)
            print_(report, file)

        macro_avg_results_df = pd.DataFrame.from_dict(
            macro_avg_results, orient='index')
        weighted_avg_results_df = pd.DataFrame.from_dict(
            weighted_avg_results, orient='index')

        with open('tests/{}_weightedAvg.md'.format(test_name), 'w') as fid:
            print_(weighted_avg_results_df.to_markdown(), fid)

        with open('tests/{}_macroAvg.md'.format(test_name), 'w') as fid:
            print_(macro_avg_results_df.to_markdown(), fid)
    else:
        for column in category_names:
            report = classification_report(Y_test[column], Y_pred_df[column])

            print('Report for {} category'.format(column))
            print(report)


def print_(text, file):
    if file:
        print(text, file=file)
    else:
        print(text)


def save_model(model, model_filepath):
    print('\nSaving model to {}\n'.format(model_filepath))

    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3 or len(sys.argv) == 5:
        start_time = time.time()

        file = None
        test_name = None
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:]
        elif len(sys.argv) == 5:
            database_filepath, model_filepath, test_description, test_name = sys.argv[1:]
            file = open('tests/{}_details.txt'.format(test_name), 'w')
            print_('Test description: {}\n'.format(test_description), file)

        X, Y, category_names = load_data(database_filepath, file)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.33)
        
        model = build_model(file)

        print('\nTraining the model...\n')

        model.fit(X_train, Y_train)

        print_('\nThe best parameters across ALL searched params: {}\n'.format(model.best_params_), file)

        evaluate_model(model, X_test, Y_test, category_names, test_name, file)

        save_model(model, model_filepath)

        end_time = time.time()

        print_('\nExecution time: {} minutes\n'.format((end_time - start_time) / 60), file)

        if file:
            print('\nExecution time: {} minutes\n'.format((end_time - start_time) / 60))

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument, the filepath of the pickle file to '
              'save the model to as the second argument, the test name as ' 
              'the third argument (optional) and the test description as the fourth '
              'argument (optional).\n\nExample:\npython train_classifier.py ' 
              '../data/DisasterResponse.db classifier.pkl "Test using RandomSearch" '
              'testRandomSearch\npython train_classifier.py ' 
              '../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
