# Disaster Response Pipeline Project
Messages classification model and web app

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - Add the project root to PYTHONPATH environment variable
        `#Example MacOS: .zshrc`
        `export PYTHONPATH=$PYTHONPATH:/Users/stefaniamindoiu/Workspace/Data/Project_Classify_Messages/DisasterMessagesClassification`
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Project structure
- /app
    - Flask web server and related files
- /data
    - csv data files
    - ETL script for reading, processing and saving the data into a database
    - sqlite database file
- /models
    - custom_transformers folder
    - script for building, training and saving the model into classifier.pkl file
- /tests
    - the results of the tests done for obtaining the model with the best results

### Cleanup
After creating single columns for each category, and populating them with the appropriat numeric values, three issues were encountered:
1. Besides values 0 and 1, 'related' column has an additional value: 2

Fix: The cells with value 2 were replaced with value 1

2. 'child_alone' column only contains 0s

Fix: Drop the column because it has no relevance

3. There are some duplicated rows

Fix: Drop them

### Notes
Several tests have been done in order to train the best model. The results of the tests can be found on /tests folder.
1. testRandomSearch1

GridSearch; RandomForestClassifier; TfidfVectorizer; no custom transformers

Parameters: {}

2. testRandomSearch2
GridSearch; RandomForestClassifier; TfidfVectorizer; NounProportion, WordsCount, CapitalWordsCount transformers

Parameters: {}

Similar results to testRandomSearch1

3. testRandomSearch3
GridSearch; RandomForestClassifier; TfidfVectorizer; NounProportion, WordsCount, CapitalWordsCount transformers; Parameters

Parameters: {
    'clf__estimator__n_estimators': [100, 200],
    'clf__estimator__min_samples_split': [2, 5, 10],
    'clf__estimator__max_depth': [10, 50, None],
    'clf__estimator__bootstrap': [True, False],
}

The best parameters across ALL searched params: {'clf__estimator__bootstrap': False, 'clf__estimator__max_depth': None, 'clf__estimator__min_samples_split': 2, 'clf__estimator__n_estimators': 200}

The results are in general better that in testRandomSearch2, but not significantly.

4. testKNeighbors4
GridSearch; KNeighbors; TfidfVectorizer; no custom transformers; no Parameters

The results are much more poorer than in testRandomSearch1

5. testKNeighbors5
GridSearch; KNeighbors; TfidfVectorizer; NounProportion, WordsCount, CapitalWordsCount transformers; no Parameters

The results are better for KNeighbors using the custom transformers

Personal obervations:
1. Training data set is imbalanced
As it can be seen in the barchart, most of the data are in "related" and 'aid_related' categories. Also, the categories containing the majority of the data ('related', 'aid_related', 'direct_report', 'request', 'other_aid') have very subjective and vague names so it is unclear for me the type of the help the people need. The model is biased by 'related' category.

