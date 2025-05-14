import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_parquet('../data/processed/winemag.parquet')
df = df[df['designation'] != 'Non-designated']

df_designation_count = df.groupby('designation').size().reset_index(name='count')
df_designation_count = df_designation_count.sort_values('count', ascending=False).head(20)

max_count = df_designation_count['count'].max()

fig = px.bar(
    df_designation_count,
    x='designation',
    y='count',
    title='Quantidade de descrições por designação',
    labels={'designation': 'Designação', 'count': 'Quantidade'}
)

fig.update_layout(
    xaxis=dict(tickangle=-45),
    yaxis=dict(range=[0, max_count])
)

st.plotly_chart(fig)

df = df[df['province'] != 'Non-province']

df_province_count = df.groupby('province').size().reset_index(name='count')
df_province_count = df_province_count.sort_values('count', ascending=False).head(25)

max_count = df_province_count['count'].max()

fig = px.bar(
    df_province_count,
    x='province',
    y='count',
    title='Quantidade de descrições por estado',
    labels={'province': 'Estado', 'count': 'Quantidade'}
)

fig.update_layout(
    xaxis=dict(tickangle=-45),
    yaxis=dict(range=[0, max_count])
)

st.plotly_chart(fig)

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
st.subheader("Distribuição de preços por país")

# Filtrando dados sem país e com preços válidos
df_price_country = df[(df['country'].notna()) & (df['price'].notna())]

# Criando o boxplot para países com mais amostras
top_countries = df_price_country['country'].value_counts().nlargest(15).index.tolist()
df_filtered = df_price_country[df_price_country['country'].isin(top_countries)]

fig = px.box(
    df_filtered,
    x='country',
    y='price',
    title='Distribuição de preços por país',
    labels={'country': 'País', 'price': 'Preço ($)'}
)

fig.update_layout(
    xaxis=dict(tickangle=-45),
    yaxis=dict(title='Preço ($)'),
)

st.plotly_chart(fig)

text = " ".join(df["description"].dropna())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

st.subheader("Nuvem de Palavras das Descrições de Vinhos")
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)
