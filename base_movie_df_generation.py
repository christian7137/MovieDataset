import pandas as pd
import numpy as np
import sklearn as sk
import math
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from matplotlib.pyplot import imshow

# Read the csv files and index appropriately.
keywords_df = pd.read_csv('./data/keywords.csv')
credits_df = pd.read_csv('./data/credits.csv')
metadata_df = pd.read_csv('./data/movies_metadata.csv')

# These three files seem to be the ones we want. Indexed on ID, we could probably combine them now.
keywords_df = keywords_df.set_index('id')
credits_df = credits_df.set_index('id')
metadata_df = metadata_df.set_index('id')
# Join the useful data frames together
temp = keywords_df.join(credits_df)
movie_df = metadata_df.join(temp)

# Here I'm just doing some of the same stuff "The Story of Film" did on Kaggle.
# This all makes sense, it's pretty standard.
movie_df = movie_df.drop(['imdb_id'], axis=1)
movie_df = movie_df.drop(['original_title'], axis=1)
movie_df = movie_df.drop(['video'], axis=1)
movie_df = movie_df.drop(['vote_count'], axis=1)
base_poster_url = 'http://image.tmdb.org/t/p/w185'
movie_df['poster_path'] = base_poster_url + movie_df['poster_path']

# Clean up from https://www.kaggle.com/hadasik/movies-analysis-visualization-newbie
def get_values(data_str):
    if isinstance(data_str, float):
        pass
    else:
        values = []
        data_str = ast.literal_eval(data_str)
        if isinstance(data_str, list):
            for k_v in data_str:
                values.append(k_v['name'])
            return values
        else:
            return None
			
print("Cleaning up JSON objects . . . may take a while")
movie_df[['genres']] = movie_df[['genres']].applymap(get_values)
movie_df[['production_companies', 'production_countries']] = movie_df[['production_companies', 'production_countries']].applymap(get_values)
movie_df[['spoken_languages', 'keywords']] = movie_df[['spoken_languages', 'keywords']].applymap(get_values)
movie_df[['cast', 'crew']] = movie_df[['cast', 'crew']].applymap(get_values)
# This takes a while to run.

# Make the Collection feature readable. The previously defined function doesn't seem to work for this feature.
temp = movie_df[movie_df['belongs_to_collection'].notnull()]
temp = temp['belongs_to_collection'].apply(ast.literal_eval).apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)
for index in temp.index:
    movie_df.loc[index, 'belongs_to_collection'] = temp.loc[index]
movie_df['belongs_to_collection'].fillna(0, inplace=True)

# I took care of some of the NaN values already.
movie_df = movie_df[pd.notnull(movie_df['title'])]
movie_df = movie_df[pd.notnull(movie_df['keywords'])]
movie_df = movie_df[pd.notnull(movie_df['original_language'])]
movie_df = movie_df[pd.notnull(movie_df['status'])]
movie_df = movie_df[pd.notnull(movie_df['release_date'])]
movie_df = movie_df[pd.notnull(movie_df['poster_path'])]
movie_df = movie_df[pd.notnull(movie_df['overview'])]
movie_df = movie_df[pd.notnull(movie_df['runtime'])]

# Make the homepage feature a binary variable since we don't care what the URL is.
movie_df['homepage'].fillna(0, inplace=True)
movie_df.loc[movie_df['homepage'] != 0, 'homepage'] = 1
movie_df['homepage'] = movie_df['homepage'].astype(np.int64)

# Replace tagline NaNs
movie_df['tagline'].fillna(0, inplace=True)
# Drop status when not released.
movie_df = movie_df.loc[movie_df['status'] == 'Released']
movie_df = movie_df.drop(['status'], axis = 1)

# Take care of budget, revenue
#print(movie_df.shape)
movie_df = movie_df.loc[movie_df['revenue'] != 0]
movie_df = movie_df.loc[movie_df['budget'] != 0]
#print(movie_df.shape)
movie_df['revenue'] = movie_df['revenue'].apply(lambda x: x if x > 100 else x*1000000)
movie_df['revenue'] = movie_df['revenue'].apply(lambda x: x if (x > 999) else x*1000)
movie_df['budget'] = movie_df['budget'].apply(lambda x: x if x > 100 else x*1000000)
movie_df['budget'] = movie_df['budget'].apply(lambda x: x if (x > 999) else x*1000)

# Check for NaNs. None, that's good.
print("NaNs in movie_df?", movie_df.isnull().any().any())
print(movie_df.describe())
print("\n\n")

# Modified from https://zeevgilovitz.com/detecting-dominant-colours-in-python
def compare(title, image, color_tuple):
    image = Image.new("RGB", (200, 200,), color_tuple)
    return image

def most_frequent_color(image):
    w, h = image.size
    pixels = image.getcolors(w * h)

    most_frequent_pixel = pixels[0]

    for count, color in pixels:
        if count > most_frequent_pixel[0]:
            most_frequent_pixel = (count, color)

    return most_frequent_pixel

def get_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def get_most_freq_c(url):
    image = get_image(url)
    color = most_frequent_color(image)
    return color[1]

print("Getting the most frequent color in all posters . . .")
print("This takes a long time too.")
movie_df['most_freq_color'] = movie_df['poster_path'].apply(get_most_freq_c)
# Takes a long time to run.
print("\n")
# Take Color from tuple to individual RGB columns
temp = movie_df['most_freq_color'].apply(pd.Series)
#print(temp)
temp = temp.drop(3, axis=1)
movie_df = pd.concat([movie_df, temp], axis=1)
movie_df = movie_df.rename(columns = {0:'red'})
movie_df = movie_df.rename(columns = {1:'green'})
movie_df = movie_df.rename(columns = {2:'blue'})
movie_df = movie_df.drop('most_freq_color', axis=1)
movie_df['green'].fillna(0, inplace=True)
movie_df['blue'].fillna(0, inplace=True)
movie_df['red'] = movie_df['red'].astype(np.int64)
movie_df['green'] = movie_df['green'].astype(np.int64)
movie_df['blue'] = movie_df['blue'].astype(np.int64)

## ONE HOT ENCODE GENRES ##
test = pd.get_dummies(movie_df['genres'].apply(pd.Series).stack()).sum(level=0)
dropping = []
for col in test.columns:
    if test[col].sum() < 50:
        dropping.append(col)
# Dropping TV movie from the list of genres, there's only one movie in here with it.
test = test.drop(dropping, axis=1)
print("New Features (Genres):", test.shape)
#print(test.columns)

## MERGE TEST INTO MOVIE_DF
movie_df = pd.concat((movie_df, test), axis = 1)
movie_df = movie_df.drop('genres', axis = 1)
print("movie_df with Genres: ", movie_df.shape)

# We saw that the spoken languages feature contained lists with far more languages than was reasonable.
# Went to an English:y/n? column.
def hasEnglish(mylist):
    if 'English' in mylist:
        return 1
    else: 
        return 0

movie_df = movie_df.rename(columns = {'spoken_languages':'english'})
movie_df['english'] = movie_df['english'].apply(hasEnglish)

# We saw that the spoken languages feature contained lists with far more languages than was reasonable.
# Went to an English:y/n? column.
def inUS(mylist):
    if 'United States of America' in mylist:
        return 1
    else: 
        return 0

movie_df = movie_df.rename(columns = {'production_countries':'produced_in_us'})
movie_df['produced_in_us'] = movie_df['produced_in_us'].apply(inUS)

# We saw that the original languages feature contained lists with far more languages than was reasonable.
# Went to an English:y/n? column.
def hasoriginal(string):
    if string == 'en':
        return 1
    else: 
        return 0

movie_df = movie_df.rename(columns = {'original_language':'originally_english'})
movie_df['originally_english'] = movie_df['originally_english'].apply(hasoriginal)

# Take care of NaNs generated by the get_dummies functions
for column in movie_df.columns:
    movie_df[column].fillna(0, inplace=True)
movie_df.loc[:,'Action':] = movie_df.loc[:,'Action':].astype(np.int64)

#print(movie_df.shape)
for index, row in movie_df.iterrows():
    if row['Action':].sum() == 0:
        movie_df = movie_df.drop(index, axis = 0)
#print(movie_df.shape)
# 11 movies dropped that had no genres.

### DEFINE FUNCTIONS ###

def makeCSV(df, filename):
    df.to_csv(('./data/' + filename), index = True, sep=',', encoding='utf-8')
	
def get_color(movieid):
    color = tuple(movie_df.loc[movieid, 'red':'blue'])
    return color

def print_color(movieid):
    color = get_color(movieid)
    image = Image.new("RGB", (200, 200,), color)
    imshow(image)
    plt.show()
    return color
	

