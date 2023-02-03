import streamlit as st
import pandas as pd
import numpy as np

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home','Dataframe','Matrix','Item-Similarity','Recomendation')
)

if option == 'Home' or option == '':
    st.write("""# SISTEM REKOMENDASI HOTEL""") #menampilkan halaman utama
elif option == 'Dataframe':
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe

    #membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    ratings=pd.read_csv('rating.csv', sep=';')
    hotels = pd.read_csv('hotel.csv', sep=';')

    df = pd.merge(ratings, hotels, on='hotelId', how='inner')
    df #menampilkan dataframe

elif option == 'Matrix':
    st.write("""## User-Item Matrix""") #menampilkan judul halaman Matrix

    ratings=pd.read_csv('rating.csv', sep=';')
    hotels = pd.read_csv('hotel.csv', sep=';')

    # Keep the hotels with over 1 ratings
    df = pd.merge(ratings, hotels, on='hotelId', how='inner')
    agg_ratings = df.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()
    agg_ratings_1 = agg_ratings[agg_ratings['number_of_ratings']>1]
    agg_ratings_1.sort_values(by='number_of_ratings', ascending=False)
    df_2 = pd.merge(df, agg_ratings_1[['namahotel']], on='namahotel', how='inner')
    matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')
    matrix #menampilkan matrix data

elif option == 'Item-Similarity':
    st.write("""## Similarity""") #menampilkan judul halaman similarity

    ratings=pd.read_csv('rating.csv', sep=';')
    hotels = pd.read_csv('hotel.csv', sep=';')

    # Keep the hotels with over 1 ratings
    df = pd.merge(ratings, hotels, on='hotelId', how='inner')
    agg_ratings = df.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()
    agg_ratings_1 = agg_ratings[agg_ratings['number_of_ratings']>1]
    agg_ratings_1.sort_values(by='number_of_ratings', ascending=False)
    df_2 = pd.merge(df, agg_ratings_1[['namahotel']], on='namahotel', how='inner')
    matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')
    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)
    item_similarity = matrix_norm.T.corr()
    item_similarity #menampilkan similarity


elif option == 'Recomendation':
    st.write("""## """) #menampilkan judul halaman similarity

    ratings=pd.read_csv('rating.csv', sep=';')
    hotels = pd.read_csv('hotel.csv', sep=';')
    # Keep the hotels with over 1 ratings
    df = pd.merge(ratings, hotels, on='hotelId', how='inner')
    agg_ratings = df.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()
    agg_ratings_1 = agg_ratings[agg_ratings['number_of_ratings']>1]
    agg_ratings_1.sort_values(by='number_of_ratings', ascending=False)
    df_2 = pd.merge(df, agg_ratings_1[['namahotel']], on='namahotel', how='inner')
    matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')
    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)
    item_similarity = matrix_norm.T.corr()
    # Pick a user ID
    picked_userid = userId
    # Pick a hotels
    picked_hotel = namahotel
    # Hotels that the target user has rating
    picked_userid_rating = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')                          .sort_values(ascending=False))                          .reset_index()                          .rename(columns={1:'rating1', 2: 'rating2', 3: 'rating3', 4: 'rating4'})
    # Similarity score of the movie American Pie with all the other movies
    picked_hotel_similarity_score = item_similarity[[picked_hotel]].reset_index().rename(columns={'namahotel':'similarity_score'})

    st.title('Dapatkan Hotel Pilihan Anda!')

    userId = st.text_input('input userId')

    namahotel = st.text_input('input namahotel')

    if st.button('Lihat Hasil Rekomendasi'):

        
