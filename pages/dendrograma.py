import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Análise de Vinhos", layout="wide")
st.title("Dendrograma de Texto - Agrupamentos de Vinhos")

@st.cache_data
def load_data():
    df = pd.read_csv('winemag-data_first150k.csv')
    df = df.dropna(subset=['description', 'country', 'variety', 'points'])
    return df

df = load_data()

with st.sidebar:
    group_field = st.selectbox("Agrupar por:", options=['country', 'variety'], index=0)
    min_count = st.slider("Mínimo de vinhos por grupo:", 10, 500, 50, 10)
    top_n = st.slider("Número de grupos (top N):", 2, 20, 8)
    keyword = st.text_input("Filtrar por palavra-chave:", "")

if keyword:
    df = df[df['description'].str.contains(keyword, case=False, na=False)]
    if df.empty:
        st.error("Nenhum vinho encontrado com essa palavra-chave.")
        st.stop()

counts = df[group_field].value_counts()
eligible = counts[counts >= min_count].index[:top_n]
if len(eligible) < 2:
    st.error("Não há grupos suficientes com o número mínimo de vinhos. Ajuste os filtros.")
    st.stop()

profiles = {}
for grp in eligible:
    texts = df.loc[df[group_field] == grp, 'description']
    sample_texts = texts.sample(min(len(texts), 200), random_state=42)
    profiles[grp] = ' '.join(sample_texts)

vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(profiles.values())

Z = linkage(X.toarray(), method='ward')

fig, ax = plt.subplots(figsize=(12, 8))
dendrogram(Z, labels=list(profiles.keys()), orientation='left', leaf_font_size=12, color_threshold=0.7 * max(Z[:, 2]))
plt.title(f"Dendrograma: agrupamento por '{group_field}'{' com filtro de ' + keyword if keyword else ''}")
plt.xlabel("Distância (Ward)")
plt.tight_layout()

col1, col2 = st.columns([3, 1])
with col1:
    st.pyplot(fig)
with col2:
    st.subheader("Estatísticas")
    st.write(f"Agrupado por: {group_field}")
    st.write(f"Grupos exibidos: {len(eligible)}")
    st.write(f"Total de descrições após filtros: {len(df)}")
    st.write("\n".join([f"- {grp}: {counts[grp]} vinhos" for grp in eligible]))
