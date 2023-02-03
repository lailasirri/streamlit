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

    st.title('Dapatkan Hotel Pilihan Anda!')

    userId = st.text_input('input userId')

    namahotel = st.text_input('input namahotel')

    # membuat tombol untuk prediksi
    if st.button('Look For Hotel Recommendations'):
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
        
        picked_userid_rating = pd.DataFrame(matrix_norm[userId].dropna(axis=0, how='all')                          .sort_values(ascending=False))                          .reset_index()                          .rename(columns={1:'rating1', 2: 'rating2', 3: 'rating3', 4: 'rating4'})
        picked_hotel_similarity_score = item_similarity[[userId]].reset_index().rename(columns={'namahotel':'similarity_score'})
        picked_userid_rating_similarity = pd.merge(left=picked_userid_rating, 
                                            right=picked_hotel_similarity_score, 
                                            on='namahotel', 
                                            how='inner')\
                                     .sort_values('similarity_score', ascending=False)[:10]
        predicted_rating = round(np.average(picked_userid_rating_similarity['rating1'], 
                                    weights=picked_userid_rating_similarity['similarity_score']), 2)
        print(f'The predicted rating for {namahotel} by user {userId} is {predicted_rating}' )

        # Item-based recommendation function
    def item_based_rec(userId, number_of_similar_items=3, number_of_recommendations =10):
        import operator
        # Hotels that the target user has not rating
        picked_userid_unrating = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
        picked_userid_unrating = picked_userid_unrating[picked_userid_unrating[userId]==True]['namahotel'].values.tolist()
        # Hotels that the target user has rating
        picked_userid_rating = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')                            .sort_values(ascending=False))                            .reset_index()                            .rename(columns={1:'rating1', 2: 'rating2', 3: 'rating3', 4: 'rating4'})
  
        # Dictionary to save the unrating hoteland predicted rating pair
        rating_prediction ={}  
        # Loop through unrating hotels          
    for picked_hotel in picked_userid_unrating: 
        # Calculate the similarity score of the picked hotel with other hotels
        picked_hotel_similarity_score = item_similarity[[namahotel]].reset_index().rename(columns={namahotel:'similarity_score'})
        # Rank the similarities between the picked user rating hotel and the picked unrating hotel.
        picked_userid_rating_similarity = pd.merge(left=picked_userid_rating, 
                                                right=picked_hotel_similarity_score, 
                                                on='namahotel', 
                                                how='inner')\
                                        .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
        # Calculate the predicted rating using weighted average of similarity scores and the ratings from user 1
        predicted_rating = round(np.average(picked_userid_rating_similarity['rating1'], 
                                        weights=picked_userid_rating_similarity['similarity_score']), 3)
        # Save the predicted rating in the dictionary
        rating_prediction[picked_hotel] = predicted_rating
        # Return the top recommended movies
    return sorted(rating_prediction.items(), key=operator.itemgetter(1), reverse=True)[:number_of_recommendations]
    # Get recommendations
    recommended_hotel = item_based_rec(userId, number_of_similar_items=3, number_of_recommendations =10)
    recommended_hotel
