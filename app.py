import streamlit as st
import pandas as pd
import numpy as np

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home','Hotels','Recommendation')
)

if option == 'Home' or option == '':
    st.write("""# Welcome to Lombok Island Hotel Recommendation System
Sistem rekomendasi hotel pulau Lombok adalah sistem yang membantu pengguna menemukan hotel yang sesuai dengan kebutuhan dan preferensi mereka. Ini bisa dilakukan dengan menganalisis data seperti rating dan preferensi pengguna sebelumnya, lokasi, harga, dan fasilitas hotel. Pada metode item-based, sistem akan menganalisis item (hotel) yang serupa dan memberikan rekomendasi hotel berdasarkan hotel yang paling serupa dengan hotel yang pernah diterima pengguna. Metode ColLaborative Filtering menggabungkan beberapa metode lain dan memberikan rekomendasi berdasarkan analisis Collaborative dari data pengguna dan item. Dengan demikian, sistem rekomendasi hotel pulau Lombok dapat membantu pengguna menemukan hotel yang sesuai dengan kebutuhan dan preferensi mereka, mempermudah proses pemesanan, dan membantu pengguna menemukan hotel terbaik untuk dikunjungi saat berlibur di pulau Lombok.""") #menampilkan halaman utama
elif option == 'Dataframe':
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe
    
    #membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    ratings=pd.read_csv('rating.csv', sep=';')
    rating 

elif option == 'Hotels':
    st.write("""## List of Hotels in Lombok """) #menampilkan judul halaman similarity
    Namahotel=pd.read_csv('namahotel.csv', sep=';')
    Namahotel
    
elif option == 'Recommendation':
    st.write("""## Hotel Recommendation""") #menampilkan judul halaman similarity

    st.title('Get Your Preferred Hotel !')

    ratings=pd.read_csv('rating.csv', sep=';')
    # Keep the hotels with over 1 ratings
    # Aggregate by hotel
    agg_ratings = ratings.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()
    # Check popular hotels
    agg_ratings.sort_values(by='number_of_ratings', ascending=False)
    # Merge data
    df_2 = pd.merge(ratings, agg_ratings[['namahotel']], on='namahotel', how='inner')
    # Create user-item matrix
    matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')
    # Normalize user-item matrix
    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)
    # Item similarity matrix using Pearson correlation
    item_similarity = matrix_norm.T.corr()
    userId = df['userId']
    namahotel = df['namahotel']
    picked_userid = userId
    picked_hotel = namahotel
    userId = st.number_input('Enter user ID',0)
    namahotel = st.text_input('Enter nama hotel')

    if st.button('View Recommendation Result'):

        ratings=pd.read_csv('rating.csv', sep=';')
         # Keep the hotels with over 1 ratings
         # Aggregate by hotel
         agg_ratings = ratings.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()
        # Check popular hotels
        agg_ratings.sort_values(by='number_of_ratings', ascending=False)
        # Merge data
        df_2 = pd.merge(ratings, agg_ratings[['namahotel']], on='namahotel', how='inner')
        # Create user-item matrix
        matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')
        # Normalize user-item matrix
        matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 0)
        # Item similarity matrix using Pearson correlation
        item_similarity = matrix_norm.T.corr()

        # Pick a user ID
        picked_userid = userId
        # Pick a hotels
        picked_hotel = namahotel
        # Hotels that the target user has rating
        picked_userid_rating = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all').sort_values(ascending=False)).reset_index().rename(columns={picked_userid:'rating'})
        # Similarity score hotels
        picked_hotel_similarity_score = item_similarity[[picked_hotel]].reset_index().rename(columns={picked_hotel:'similarity_score'})
        n = 10
        picked_userid_rating_similarity = pd.merge(left=picked_userid_rating, 
                                                    right=picked_hotel_similarity_score, 
                                                    on='namahotel', 
                                                    how='inner')\
                                             .sort_values('similarity_score', ascending=False)[:10]
        predicted_rating = round(np.average(picked_userid_rating_similarity['rating'], 
                                    weights=picked_userid_rating_similarity['similarity_score']), 6)
        print(f'The predicted rating for {picked_hotel} by user {picked_userid} is {predicted_rating}' )
        
        # Item-based recommendation function
        def item_based_rec(picked_userid = userId, picked_hotel = namahotel, number_of_similar_items=5, number_of_recommendations =5):
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
             # Remove entries with NaN values
              rating_prediction = {k: v for k, v in rating_prediction.items() if not math.isnan(v)}
             # Return the top recommended movies
            return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]
        # Get recommendations
        recommended_hotel = item_based_rec(picked_userid = userId, picked_hotel = namahotel, number_of_similar_items=5, number_of_recommendations =5)
 
        st.success(recommended_hotel)
