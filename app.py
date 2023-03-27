# Data processing
import pandas as pd
import numpy as np
# Streamlit
import streamlit as st
from PIL import Image
#
import operator
import math

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home','Hotels','Recommendation')
)

if option == 'Home' or option == '':
    st.write(""" # Welcome to Recomendation System of Lombok Island Hotel""")
    st.image('hotellombok.jpeg')
    st.write("""
Sistem rekomendasi hotel pulau Lombok adalah sistem yang membantu pengguna menemukan hotel yang sesuai dengan kebutuhan dan preferensi mereka. Ini bisa dilakukan dengan menganalisis data seperti rating dan preferensi pengguna sebelumnya, lokasi, harga, dan fasilitas hotel. Pada metode item-based, sistem akan menganalisis item (hotel) yang serupa dan memberikan rekomendasi hotel berdasarkan hotel yang paling serupa dengan hotel yang pernah diterima pengguna. Metode ColLaborative Filtering menggabungkan beberapa metode lain dan memberikan rekomendasi berdasarkan analisis Collaborative dari data pengguna dan item. Dengan demikian, sistem rekomendasi hotel pulau Lombok dapat membantu pengguna menemukan hotel yang sesuai dengan kebutuhan dan preferensi mereka, mempermudah proses pemesanan, dan membantu pengguna menemukan hotel terbaik untuk dikunjungi saat berlibur di pulau Lombok.""") #menampilkan halaman utama
elif option == 'Dataframe':
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe
    
    #membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    ratings=pd.read_csv('datafix.csv', sep=';')
    rating 

elif option == 'Hotels':
    st.write("""## List of Hotels in Lombok """) #menampilkan judul halaman similarity
    Namahotel=pd.read_csv('datahotelbintang.csv', sep=';')
    Namahotel
    
elif option == 'Recommendation':
    st.write("""## Hotel Recommendation""") #menampilkan judul halaman similarity

    st.title('Get Your Preferred Hotel !')

    ratings=pd.read_csv('ap.csv', sep=';')
    # Aggregate by hotel
    agg_ratings = ratings.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()

    # Check popular hotels
    agg_ratings.sort_values(by='number_of_ratings', ascending=False)

    # Merge data
    df_2 = pd.merge(ratings, agg_ratings[['namahotel']], on='namahotel', how='inner')
    df_2.info()

    # Create user-item matrix
    matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')

    # Normalize user-item matrix
    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)

    # Item similarity matrix using Pearson correlation
    item_similarity = matrix_norm.T.corr()
    namahotel = st.text_input('Enter nama hotel')
   

    if st.button('View Recommendation Result'):

            ratings=pd.read_csv('apa.csv', sep=';')
            # Aggregate by hotel
            agg_ratings = ratings.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()

            # Check popular hotels
            agg_ratings.sort_values(by='number_of_ratings', ascending=False)

            # Merge data
            df_2 = pd.merge(ratings, agg_ratings[['namahotel']], on='namahotel', how='inner')
            df_2.info()

            # Create user-item matrix
            matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')

            # Normalize user-item matrix
            matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)

            # Item similarity matrix using Pearson correlation
            item_similarity = matrix_norm.T.corr()

            # Pick a hotel
            picked_hotel = namahotel

            # Similarity score of the hotel with all the other hotels
            picked_hotel_similarity_score = item_similarity[[picked_hotel]].reset_index().rename(columns={namahotel:'similarity_score'})

            # Rank the similarities between the hotels and the picked hotel.
            n=10
            picked_hotel_similarity = pd.merge(left=agg_ratings[['namahotel']], 
                                                right=picked_hotel_similarity_score, 
                                                on='namahotel', 
                                                how='inner')\
                                        .sort_values('similarity_score', ascending=False)[:n]


            st.table(picked_hotel_similarity)
