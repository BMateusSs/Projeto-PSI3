import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_parquet('../data/processed/winemag.parquet')
df = df[df['designation'] != 'Non-designated']
text = " ".join(df["description"].dropna())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

st.subheader("Nuvem de Palavras das Descrições de Vinhos")
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)