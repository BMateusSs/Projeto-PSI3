import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_parquet('data/processed/winemag.parquet')
df = df[df['designation'] != 'Non-designated']

df_designation_count = df.groupby('designation').size().reset_index(name='count')
df_designation_count = df_designation_count.sort_values('count', ascending=False).head(20)

max_count = df_designation_count['count'].max()

fig = px.bar(
    df_designation_count,
    x='designation',
    y='count',
    title='Quantidade de descrições por `designation`',
    labels={'designation': 'Designation', 'count': 'Quantidade'}
)

fig.update_layout(
    xaxis=dict(tickangle=-45),
    yaxis=dict(range=[0, max_count])
)

st.plotly_chart(fig)