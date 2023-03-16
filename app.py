# Data processing
import pandas as pd
import numpy as np
import scipy.stats 
# Visualization
import seaborn as sns
# Json
import json
# Streamlit
import streamlit as st
from surprise import accuracy
import operator
import math

# Read in data
ratings=pd.read_csv('datafix.csv', sep=';')
ratings

# Aggregate by hotel
agg_ratings = ratings.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()

# Check popular hotels
agg_ratings.sort_values(by='number_of_ratings', ascending=False)

# Merge data
df_2 = pd.merge(ratings, agg_ratings[['namahotel']], on='namahotel', how='inner')
df_2.info()

# Create user-item matrix
matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')
matrix

# Normalize user-item matrix
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)
matrix_norm

# Item similarity matrix using Pearson correlation
item_similarity = matrix_norm.T.corr()
item_similarity

# Pick a user ID
picked_userid = 7
# Pick a hotels
picked_hotel = 'ASTON Inn Mataram'
# Hotels that the target user has rating
picked_userid_rating = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')\
                          .sort_values(ascending=False))\
                          .reset_index()\
                          .rename(columns={7:'rating'})
picked_userid_rating

# Similarity score of the movie American Pie with all the other movies
picked_hotel_similarity_score = item_similarity[[picked_hotel]].reset_index().rename(columns={'ASTON Inn Mataram':'similarity_score'})

# Rank the similarities between the hotels user 1 rated and Aston Inn Mataram.
n=10
picked_userid_rating_similarity = pd.merge(left=picked_userid_rating, 
                                            right=picked_hotel_similarity_score, 
                                            on='namahotel', 
                                            how='inner')\
                                     .sort_values('similarity_score', ascending=False)[:10]
# Take a look at the User 1 watched movies with highest similarity
picked_userid_rating_similarity

# Calculate the predicted rating using weighted average of similarity scores and the ratings from user 
predicted_rating = round(np.average(picked_userid_rating_similarity['rating'], 
                                    weights=picked_userid_rating_similarity['similarity_score']), 6)
print(f'The predicted rating for {picked_hotel} by user {picked_userid} is {predicted_rating}' )

def item_based_rec(picked_userid, number_of_similar_items=5, number_of_recommendations =5):
  # Hotels that the target user has not rated
  picked_userid_unrating = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
  picked_userid_unrating = picked_userid_unrating[picked_userid_unrating[picked_userid]==True]['namahotel'].values.tolist()
  
  # Hotels that the target user has rated
  picked_userid_rating = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')\
                            .sort_values(ascending=False))\
                            .reset_index()\
                            .rename(columns={7:'rating'})
    
  # Check if the user has rated any hotels
  if picked_userid_rating.empty:
    # If the user has not rated any hotels, recommend the most popular hotels
    most_popular_hotels = matrix_norm.mean(axis=1).sort_values(ascending=False).index.tolist()
    return most_popular_hotels[:number_of_recommendations]
  
  # Dictionary to save the unrated hotel and predicted rating pair
  rating_prediction ={}  
  
  # Loop through unrated hotels          
  for picked_hotel in picked_userid_unrating: 
    # Calculate the similarity score of the picked hotel with other hotels
    picked_hotel_similarity_score = item_similarity[[picked_hotel]].reset_index().rename(columns={picked_hotel:'similarity_score'})
    
    # Rank the similarities between the picked user rated hotel and the picked unrated hotel.
    picked_userid_rating_similarity = pd.merge(left=picked_userid_rating, 
                                                right=picked_hotel_similarity_score, 
                                                on='namahotel', 
                                                how='inner')\
                                        .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
    
    # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
    predicted_rating = round(np.average(picked_userid_rating_similarity['rating'], 
                                        weights=picked_userid_rating_similarity['similarity_score']), 6)
    
    # Save the predicted rating in the dictionary
    rating_prediction[picked_hotel] = predicted_rating
    
  # Remove entries with NaN values
    rating_prediction = {k: v for k, v in rating_prediction.items() if not math.isnan(v)}
  
  # Return the top recommended hotels
  return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]

# Get recommendations
recommended_hotel = item_based_rec(picked_userid, number_of_similar_items=5, number_of_recommendations=10)
recommended_hotel
