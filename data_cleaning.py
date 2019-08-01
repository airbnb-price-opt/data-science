import pandas as pd
import numpy as np
import os
import pathlib


def clean_zips(df):
    """
    A helper function to clean zip code columns
    for Airbnb Data
    """

    def first_ele(df):
        for i in df:
            return i

    def first_five(df):
        return df[2:7]

    # Splits data for entries containing period
    df['zipcode'] = df['zipcode'].str.split('.')

    # Transform all values into strings
    df['zipcode'] = df['zipcode'].apply(str)

    # Returns only the first five characters
    df['zipcode'] = df['zipcode'].map(first_five)

    df = df.loc[df['zipcode'] != 'n']

    return df


def clean_price(df):
    df['price'] = df['price'].str.strip('$')
    df['cleaning_fee'] = df['cleaning_fee'].str.strip('$')

    df['price'] = df['price'].str.replace(',', '')
    df['cleaning_fee'] = df['cleaning_fee'].str.replace(',', '')

    df['cleaning_fee'] = df['cleaning_fee'].replace(np.nan, 0)

    df['price'] = df['price'].astype(float)
    df['cleaning_fee'] = df['cleaning_fee'].astype(float)

    df['total_price'] = df['price'] + df['cleaning_fee']
    df = df.drop(['price', 'cleaning_fee'], axis=1)

    df = df[df['total_price'] > 1]

    # Log transform total price
    df['price_log'] = df['total_price'].apply(lambda x: np.log(x))

    return df


def clean_data(df):
    # Drop columns containing null values
    df = df.dropna()

    return df


def data_cleaning():
    print("Info: Step 2 - Data Cleaning start ...")

    # Remove old data if present
    file = pathlib.Path("data-clean.csv")
    if file.exists():
        print("Info: Removing data-clean.csv")
        os.remove("data-clean.csv")

    df = pd.read_csv('data.csv')
    df = clean_data(clean_price(clean_zips(df)))
    df.to_csv('data-clean.csv', index=False)

    print("Info: Data Cleaning completed ...")

if __name__ == '__main__':
    data_cleaning()