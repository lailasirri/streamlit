import streamlit as st
import pandas as pd
import numpy as np

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

elif option == 'Matrix':
    st.write("""## Draw User-Item Matrix""") #menampilkan judul halaman 
# Create user-item matrix
matrix = df_2.pivot_table(index='namahotel', columns='userId', values='rating')
matrix


    
