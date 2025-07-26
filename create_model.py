import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load the data
print("Loading CSV files...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge the datasets
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# Remove rows with missing data
movies.dropna(inplace=True)

# Function to convert text to list of names
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L

# Function to get first 3 cast members
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L

# Function to get director
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Function to collapse spaces
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

print("Processing genres...")
movies['genres'] = movies['genres'].apply(convert)

print("Processing keywords...")
movies['keywords'] = movies['keywords'].apply(convert)

print("Processing cast...")
movies['cast'] = movies['cast'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])

print("Processing crew...")
movies['crew'] = movies['crew'].apply(fetch_director)

print("Collapsing spaces...")
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

print("Processing overview...")
movies['overview'] = movies['overview'].apply(lambda x:x.split())

print("Creating tags...")
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create new dataframe with only necessary columns
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])

print("Joining tags...")
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

print("Creating feature vectors...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

print("Calculating similarity matrix...")
similarity = cosine_similarity(vector)

print("Saving model files...")
# Save the files in the model directory
pickle.dump(new, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

print("Model files created successfully!")
print(f"movie_list.pkl size: {os.path.getsize('model/movie_list.pkl')} bytes")
print(f"similarity.pkl size: {os.path.getsize('model/similarity.pkl')} bytes") 