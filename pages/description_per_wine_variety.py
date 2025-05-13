import streamlit as st 
import pandas as pd
import plotly.express as px

df = pd.read_parquet('data/processed/winemag.parquet')

df = df[df['variety'].notna()]

df_variety_count = df.groupby('variety').size().reset_index(name='count')
df_variety_count = df_variety_count.sort_values('count', ascending=False).head(25)

max_count = df_variety_count['count'].max()

fig = px.bar(
    df_variety_count,
    x='variety',
    y='count',
    title='Quantidade de descrições por variedade de vinho',
    labels={'variety': 'Variedade', 'count': 'Quantidade'}
)

fig.update_layout(
    xaxis=dict(tickangle=-45),
    yaxis=dict(range=[0, max_count])
)

st.plotly_chart(fig)