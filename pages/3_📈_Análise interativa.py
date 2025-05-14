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

    min_price = int(df['price'].min())
    max_price = int(df['price'].max())
    price_range = st.slider(
        label="Faixa de preço (USD):",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )

    varieties = st.multiselect(
        "Selecione as variedades:",
        options=sorted(df['variety'].dropna().unique()),
        default=[]
    )

filtered_df = df[
    (df['country'].isin(countries)) &
    (df['points'] >= points_range[0]) & (df['points'] <= points_range[1]) &
    (df['price'] >= price_range[0]) & (df['price'] <= price_range[1])
]
