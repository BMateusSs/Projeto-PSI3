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

    min_points = int(df['points'].min())
    max_points = int(df['points'].max())
    points_range = st.slider(
        label="Faixa de pontuação:",
        min_value=min_points,
        max_value=max_points,
        value=(min_points, max_points)
    )

    #filtro por faixa de preco

    #filtro por variedades