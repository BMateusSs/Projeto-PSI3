import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_parquet('data/processed/winemag.parquet')
df = df[df['province'] != 'Non-province']

df_province_count = df.groupby('province').size().reset_index(name='count')
df_province_count = df_province_count.sort_values('count', ascending=False).head(25)

max_count = df_province_count['count'].max()

fig = px.bar(
    df_province_count,
    x='province',
    y='count',
    title='Quantidade de descrições por province',
    labels={'province': 'province', 'count': 'Quantidade'}
)

fig.update_layout(
    xaxis=dict(tickangle=-45),
    yaxis=dict(range=[0, max_count])
)

st.plotly_chart(fig)