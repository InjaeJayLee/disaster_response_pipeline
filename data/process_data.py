import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath) -> pd.DataFrame:
    """
    Load messages and categoreis and returns a merged DataFrame of them
    :param messages_filepath: messages data file path
    :param categories_filepath: categories data file path
    :return: a result of the merged DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, how='inner', on='id')


def clean_data(df) -> pd.DataFrame:
    """
    Extract categorical data in 'categories' column, let them have values of only either 0 or 1 and remove duplicate rows
    :param df: a DataFrame to clean
    :return: a result of the cleaned DataFrame
    """

    # clean categorical data
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = categories[column].astype(int).apply(lambda x: 1 if x > 1 else x)
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    return df.drop_duplicates()


def save_data(df, database_filename):
    """
    save the data into a sql db
    """
    engine = create_engine(f'sqlite:///{database_filename}.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
