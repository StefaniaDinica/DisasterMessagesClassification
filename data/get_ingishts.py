from sqlalchemy import create_engine
import pandas as pd

def main():
    '''Reads data from the database and loads it into dataframes

    Args:
    database_filepath (string) - path to the database file

    Returns:
    X - dataframe containing 'message' column;
    Y - dataframe containing all categories columns;
    category_names - list with all the categories names
    '''
    engine = create_engine('sqlite:///{}'.format('data/DisasterResponse.db'))
    df = pd.read_sql_table('MessagesCategories', con=engine.connect())

    category_names = df.drop(columns=['id', 'message', 'original', 'genre']).columns
    
    # make a bar chart
    print(df[category_names].sum().sort_values())
    print(df.shape)

    # bar chart 
    # print(df[df['genre'] == 'direct'].sum(axis=1))
    print(df['genre'].value_counts())
    print('direct')
    print(df[df['genre'] == 'direct'][category_names].sum(axis=1).value_counts().sort_index())
    print('news')
    print(df[df['genre'] == 'news'][category_names].sum(axis=1).value_counts().sort_index())
    print('social')
    print(df[df['genre'] == 'social'][category_names].sum(axis=1).value_counts().sort_index())


    print(df[category_names].sum(axis=1).value_counts().sort_index())

if __name__ == '__main__':
    main()

