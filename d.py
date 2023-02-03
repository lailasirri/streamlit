import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Home','Dataframe','Chart')
)

if option == 'Home' or option == '':
    st.write("""# Daptkan Hotel Terbaik Pilihan Anda!""") #menampilkan halaman utama
elif option == 'Dataframe':
    st.write("""## Dataframe""") #menampilkan judul halaman dataframe

    #membuat dataframe dengan pandas yang terdiri dari 2 kolom dan 4 baris data
    ratings=pd.read_csv('rating.csv', sep=';')
    hotels = pd.read_csv('hotel.csv', sep=';')

    df = pd.merge(ratings, hotels, on='hotelId', how='inner')
    df #menampilkan dataframe

    # Aggregate by hotel
    agg_ratings = df.groupby('namahotel').agg(mean_rating = ('rating', 'mean'), number_of_ratings = ('rating', 'count')).reset_index()
    
    # Keep the hotels with over 1 ratings
    agg_ratings_1 = agg_ratings[agg_ratings['number_of_ratings']>1]

    # Check popular hotels
    agg_ratings_1.sort_values(by='number_of_ratings', ascending=False)

elif option == 'Chart':
    st.write("""## Draw Charts""") #menampilkan judul halaman 

    def jointplot_chart(agg_ratings_1):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,7))
    sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_1)
    plt.show()

    st.pyplot(jointplot_chart(agg_ratings_1))

    
