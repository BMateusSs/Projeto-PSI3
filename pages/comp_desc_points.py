import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_parquet('data/processed/winemag.parquet')
df = df[df['designation'] != 'Non-designated']

df['desc_length'] = df['description'].dropna().apply(len)
fig = px.scatter(
    df,
    x='desc_length',
    y='points',
    title='Comprimento da descrição x Pontuação',
    labels={'desc_length': 'Comprimento da descrição', 'points': 'Pontuação'},
    opacity=0.6,
    trendline='ols',
    color='points'
)
st.plotly_chart(fig)
