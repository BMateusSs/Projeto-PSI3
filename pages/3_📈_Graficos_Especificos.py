import streamlit as st
import pandas as pd
import plotly.express as px

def load_data():
    return pd.read_parquet("data/processed/winemag.parquet")

df = load_data()

with st.sidebar:
    st.header("Filtros")
    countries = st.multiselect(
        "Selecione os países:",
        options=df['country'].unique(),
        default=['US', 'France', 'Italy']
    )
    #Filtro por faixa de pontuação

    #filtro por faixa de preco

    #filtro por variedades