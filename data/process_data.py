import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Reads data from two csv files, merges the data and returns the data as a dataframe
    
    Args:
    messages_filepath (string): path to messages csv file;
    categories_filepath (string): path to categories file

    Returns:
    df - pandas dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on=['id'])

    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    category_colnames = [item[:-2] for item in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda value: value[len(value) - 1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # 'related' category has 0, 1 and 2 values -> replace 2 values with 1 values
    categories['related'] = categories['related'].apply(lambda x: 1 if x == 2 else x)

    # 'child_alone' category has only 0 value -> drop column
    categories = categories.drop(columns=['child_alone'])

    # drop the original categories column from `df`
    df = df.drop(columns=["categories"])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)

    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()