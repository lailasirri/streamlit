import streamlit as st
import pandas as pd
import numpy as np

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home','Recommendation')
)

if option == 'Home' or option == '':
    st.write("""# Welcome to Lombok Island Hotel Recommendation System
Sistem rekomendasi hotel pulau Lombok adalah sistem yang membantu pengguna menemukan hotel yang sesuai dengan kebutuhan dan preferensi mereka. Ini bisa dilakukan dengan menganalisis data seperti rating dan preferensi pengguna sebelumnya, lokasi, harga, dan fasilitas hotel. Pada metode item-based, sistem akan menganalisis item (hotel) yang serupa dan memberikan rekomendasi hotel berdasarkan hotel yang paling serupa dengan hotel yang pernah diterima pengguna. Metode ColLaborative Filtering menggabungkan beberapa metode lain dan memberikan rekomendasi berdasarkan analisis Collaborative dari data pengguna dan item. Dengan demikian, sistem rekomendasi hotel pulau Lombok dapat membantu pengguna menemukan hotel yang sesuai dengan kebutuhan dan preferensi mereka, mempermudah proses pemesanan, dan membantu pengguna menemukan hotel terbaik untuk dikunjungi saat berlibur di pulau Lombok.""") #menampilkan halaman utama
elif option == 'Dataframe':
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe
    
    #membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    ratings=pd.read_csv('rating.csv', sep=';')
    hotels = pd.read_csv('hotel.csv', sep=';')

    df = pd.merge(ratings, hotels, on='hotelId', how='inner')
    df #menampilkan dataframe


elif option == 'Recommendation':
    st.write("""## Hotel Recommendation""") #menampilkan judul halaman similarity

    st.title('Get Your Preferred Hotel !')

    ratings=pd.read_csv('rating.csv', sep=';')
    hotels = pd.read_csv('hotel.csv', sep=';')
    df = pd.merge(ratings, hotels, on='hotelId', how='inner')
    userId = df['userId']
    namahotel = df['namahotel']
    picked_userid = userId
    picked_hotel = namahotel
    picked_userid = st.number_input('Enter user ID',0)
    picked_hotel = st.text_input('Enter nama hotel')

    if st.button('View Recommendation Result'):

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
        ratings=pd.read_csv('rating.csv', sep=';')
        hotels = pd.read_csv('hotel.csv', sep=';')

        # Pick a user ID
        picked_userid = 3
        # Pick a hotels
        picked_hotel = 'ASTON Inn Mataram'
        # Hotels that the target user has rating
        picked_userid_rating = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all').sort_values(ascending=False)).reset_index().rename(columns={picked_userid:'rating'})
        # Similarity score hotels
        picked_hotel_similarity_score = item_similarity[[picked_hotel]].reset_index().rename(columns={picked_hotel:'similarity_score'})
        n = 5
        picked_userid_rating_similarity = pd.merge(left=picked_userid_rating, 
                                                    right=picked_hotel_similarity_score, 
                                                    on='namahotel', 
                                                    how='inner')\
                                             .sort_values('similarity_score', ascending=False)[:5]
        predicted_rating = round(np.average(picked_userid_rating_similarity['rating'], 
                                    weights=picked_userid_rating_similarity['similarity_score']), 2)
        print(f'The predicted rating for {picked_hotel} by user {picked_userid} is {predicted_rating}' )
        
        # Item-based recommendation function
        def item_based_rec(picked_userid, picked_hotel, number_of_similar_items=3, number_of_recommendations =10):
            import operator
            # Hotels that the target user has not rating
            picked_userid_unrating = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
            picked_userid_unrating = picked_userid_unrating[picked_userid_unrating[picked_userid]==True]['namahotel'].values.tolist()
            # Hotels that the target user has rating
            picked_userid_rating = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all').sort_values(ascending=False)).reset_index().rename(columns={picked_userid:'rating'})
  
            # Dictionary to save the unrating hoteland predicted rating pair
            rating_prediction ={}  
            # Loop through unrating hotels          
            for picked_hotel in picked_userid_unrating: 
              # Calculate the similarity score of the picked hotel with other hotels
              picked_hotel_similarity_score = item_similarity[[picked_hotel]].reset_index().rename(columns={picked_hotel:'similarity_score'})
              # Rank the similarities between the picked user rating hotel and the picked unrating hotel.
              picked_userid_rating_similarity = pd.merge(left=picked_userid_rating, 
                                                          right=picked_hotel_similarity_score, 
                                                          on='namahotel', 
                                                          how='inner')\
                                                  .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
                
              # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 
              predicted_rating = round(np.average(picked_userid_rating_similarity['rating'], 
                                                  weights=picked_userid_rating_similarity['similarity_score']), 6)
             # Save the predicted rating in the dictionary
              rating_prediction[picked_hotel] = predicted_rating
             # Return the top recommended movies
            return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]
        # Get recommendations
        recommended_hotel = item_based_rec(picked_userid, picked_hotel, number_of_similar_items=3, number_of_recommendations =10)
        recommended_hotel

        st.success(recommended_hotel)
