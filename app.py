import streamlit as st
import numpy as np
import pandas as pd
import spacy
from library import predict_spacy



if __name__ == '__main__':
    laptops = pd.read_csv("Final_Dataframe.csv")
    df = pd.read_csv("final.csv")
    nlp = spacy.load("en_core_web_lg") 
    df['spacy_sentence'] = df['sentence'].apply(lambda x: nlp(x)) 
    embed_mat = df['spacy_sentence'].values
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Laptop Recommender </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    laptop_name = st.text_input("Enter Laptop name here!!","")
    best_index = predict_spacy(nlp, laptop_name, embed_mat, 8)
    
    if st.button("Search"):
        lr=laptops.iloc[best_index]
        st.write(lr)


