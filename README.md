# Disaster Response Pipeline Project
Messages classification model and web app

## Overview of the app
![Dataset overview](/assets/dataset_overview.png)

The web application is built with Flask.
The model was built using [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) algorithm.

## Overview of the training set

![Dataset overview](/assets/dataset_overview.png)

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - Install all the dependecies in `requirements.txt`
    - Add the project root to PYTHONPATH environment variable
        
        Example for MacOS (.zshrc):
        
        `export PYTHONPATH=$PYTHONPATH:/Users/stefaniamindoiu/Workspace/Data/Project_Classify_Messages/DisasterMessagesClassification`
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Project structure
- /app
    - Flask web server and related files
- /data
    - csv data files
    - ETL script for reading, processing and saving the data into a database
    - sqlite database file
- /models
    - script for building, training and saving the model into classifier.pkl file
- /tests
    - the results of the tests done for obtaining the model with the best results
- /transformers
    - custom transformers

## Dataset cleanup
After creating single columns for each category, and populating them with the appropriat numeric values, three issues were encountered:
1. Besides values 0 and 1, 'related' column has an additional value: 2

Fix: The cells with value 2 were replaced with value 1

2. 'child_alone' column only contains 0s

Fix: Drop the column because it has no relevance

3. There are some duplicated rows

Fix: Drop them

## Tests
Several tests have been done in order to train the best model. The results of the tests can be found in /tests folder.
#### 1. testRandomSearch1

GridSearch; RandomForestClassifier; TfidfVectorizer; no custom transformers; no parameters

[testRandomSearch1_macroAvg.md](/tests/testRandomSearch1_macroAvg.md)

[testRandomSearch1_weightedAvg.md](/tests/testRandomSearch1_weightedAvg.md)

#### 2. testRandomSearch2

GridSearch; RandomForestClassifier; TfidfVectorizer; NounProportion, WordsCount, CapitalWordsCount transformers

[testRandomSearch2_macroAvg.md](/tests/testRandomSearch2_macroAvg.md)

[testRandomSearch2_weightedAvg.md](/tests/testRandomSearch2_weightedAvg.md)


Similar results to testRandomSearch1

#### 3. testRandomSearch3

GridSearch; RandomForestClassifier; TfidfVectorizer; NounProportion, WordsCount, CapitalWordsCount transformers;


    parameters = {
        'clf__estimator__bootstrap': [True, False],
        'clf__estimator__max_depth': [10, 50, None],
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__n_estimators': [100, 200],
    }

The best parameters across ALL searched params:

    {
        'clf__estimator__bootstrap': False,
        'clf__estimator__max_depth': None,
        'clf__estimator__min_samples_split': 2,
        'clf__estimator__n_estimators': 200
    }

The results are in general better that in testRandomSearch2, but not significantly.

[testRandomSearch3_macroAvg.md](/tests/testRandomSearch3_macroAvg.md)

[testRandomSearch3_weightedAvg.md](/tests/testRandomSearch3_weightedAvg.md)

#### 4. testKNeighbors1

GridSearch; KNeighborsClassifier; TfidfVectorizer; no custom transformers; no parameters

The results are much more poorer than in testRandomSearch1

[testKNeighbors1_macroAvg.md](/tests/testKNeighbors1_macroAvg.md)

[testKNeighbors1_weightedAvg.md](/tests/testKNeighbors1_weightedAvg.md)

#### 5. testKNeighbors2
GridSearch; KNeighborsClassifier; TfidfVectorizer; NounProportion, WordsCount, CapitalWordsCount transformers; no parameters

The results are better for KNeighbors using the custom transformers

[testKNeighbors2_macroAvg.md](/tests/testKNeighbors2_macroAvg.md)

[testKNeighbors2_weightedAvg.md](/tests/testKNeighbors2_weightedAvg.md)

#### 6. testKNeighbors3
GridSearch; KNeighborsClassifier; TfidfVectorizer; NounProportion, WordsCount, CapitalWordsCount transformers

    parameters = {
        'clf__estimator__n_neighbors': [5, 10, 20, 30],
        'clf__estimator__leaf_size': [15, 30, 50],
        'clf__estimator__p': [1, 2]
    }

The best parameters across ALL searched params:

    {
        'clf__estimator__n_neighbors': 10,
        'clf__estimator__leaf_size': 15,
        'clf__estimator__p': 1
    }


The results are poorer than in testRandomSearch3

[testKNeighbors3_macroAvg.md](/tests/testKNeighbors3_macroAvg.md)

[testKNeighbors3_weightedAvg.md](/tests/testKNeighbors3_weightedAvg.md)

#### 7. testSVC1
GridSearch; SVC; TfidfVectorizer; no custom transformers; no parameters

The results are better than for testRandomSearch1.

[testSVC1_macroAvg.md](/tests/testSVC1_macroAvg.md)

[testSVC1_weightedAvg.md](/tests/testSVC1_weightedAvg.md)

#### 8. testSVC2
GridSearch; SVC; TfidfVectorizer; NounProportion, WordsCount, CapitalWordsCount transformers; no parameters

The results are much more poorer than in testSVC7 -> SVC is better used without the custom transformers

[testSVC2_macroAvg.md](/tests/testSVC2_macroAvg.md)

[testSVC2_weightedAvg.md](/tests/testSVC2_weightedAvg.md)

#### 9. testSVC3
GridSearch; SVC; TfidfVectorizer; no custom transformers; Parameters

    parameters = {
        'clf__estimator__degree': [2, 3, 4]
        'clf__estimator__gamma': ['scale', 'auto'],
        'clf__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    }

    The best parameters across ALL searched params:

    {
        'clf__estimator__degree': 2,
        'clf__estimator__gamma': 'scale',
        'clf__estimator__kernel': 'linear'
    }


[testSVC2_macroAvg.md](/tests/testSVC2_macroAvg.md)

[testSVC2_weightedAvg.md](/tests/testSVC2_weightedAvg.md)



## Obervations:
1. Training data set is imbalanced

As it can be seen in the barchart, most of the data are in "related" and 'aid_related' categories. Also, the categories containing the majority of the data ('related', 'aid_related', 'direct_report', 'request', 'other_aid') have very subjective and vague names so it is unclear for me the type of the help the people need. The model is biased by 'related' category.

2. The best model was obtained with **Support Vector Classifier** algorithm with the following parameters:

    {
        'clf__estimator__degree': 2,
        'clf__estimator__gamma': 'scale',
        'clf__estimator__kernel': 'linear'
    }

