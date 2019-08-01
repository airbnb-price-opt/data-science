# Web Scraping
import requests
from bs4 import BeautifulSoup
import pathlib

# Merging Data
import time
import gzip
import pandas as pd
import numpy as np
import os


def retrieve_html():
    """Function that locates the exact files needed for data extraction"""

    page = requests.get("http://insideairbnb.com/get-the-data.html")

    if page.status_code != 200:
        print("Error: Request to InsideAirBnB is failing")

    soup = BeautifulSoup(page.content, 'html.parser')

    td_tags = soup.find_all('td')

    # To ensure only the latest data for a particular city is used.
    city_set = set()

    # To maintain city level summary for data fetched.
    city_data = []

    for td_tag in td_tags:
        link_list = [a['href'] for a in td_tag.find_all('a', href=True)]

        if len(link_list) > 0 and link_list[0].find('listings.csv.gz') != -1:
            # Summary for each city is got by parsing the url itself.
            url_split = link_list[0].split('/')

            if len(url_split) != 9:
                print(f"Error: URL not following the "  # noqa: E999
                      f"format {link_list[0]}")

            if url_split[3] == 'united-states':
                country = url_split[3]
                region = url_split[4]
                city = url_split[5]
                date = url_split[6]
                url = link_list[0]

                if city not in city_set:
                    city_set.add(city)
                    city_data.append([country, region, city, date, url])

    # Check summary information of each city.
    print(f"Info: Total number of city information fetched: {len(city_data)}")
    print("Info: Start summary information of each city")
    for city in city_data:
        print(city)
    print("Info: Completed summary information of each city ")

    return city_data


def download_dataframe(city_data):
    # Consolidated data frame to hold data for all cities together.
    df_all = pd.DataFrame()

    for city in city_data:
        city_name = city[2]
        url = city[4]
        print(f"Info: Downloading data for {city_name} with url {url}")

        r = requests.get(url)

        # Retrieve HTTP meta-data.
        if r.status_code != 200:
            print(f"Error: Request to {url} failed with status "
            "{r.status_code}")  # noqa: E999
            continue

        # Fetch the data locally.
        file_name = f"{city_name}_listings.csv.gz"
        with open(file_name, 'wb') as f:
            f.write(r.content)

        # Unzip and load the file to data frame.
        with gzip.open(file_name) as f:
            df = pd.read_csv(f)

            print(f"Info: Shape of data within {file_name}: {df.shape}")

            if df_all.empty:
                df_all = df
            else:
                df_all = pd.concat([df_all, df])

            print(f"Info: Shape of concatenated dataframe: {df_all.shape}")

        # Remove file
        os.remove(file_name)
        print(f"Info: Removed {file_name}!")

        # Sleep for short duration to ensure server is not loaded
        time.sleep(10)

    return df_all


def reduce_df(df):

    columns = ['zipcode',
               'property_type',
               'room_type',
               'accommodates',
               'bathrooms',
               'bedrooms',
               'beds',
               'bed_type',
               'price',
               'cleaning_fee']
    df = df[columns]

    return df


def data_retrieval():
    print("Info: Step 1 - Data Retrieval start ...")

    # Remove old data if present
    file = pathlib.Path("data.csv")
    if file.exists():
        print("Info: Removing data.csv")
        os.remove("data.csv")

    df = reduce_df(download_dataframe(retrieve_html()))
    df.to_csv('data.csv', index=False)
    print(f"Info: Shape of dataframe after merging: {df.shape}")

    print(f"Info: Data Retrieval completed ...")

if __name__ == '__main__':
    data_retrieval()
