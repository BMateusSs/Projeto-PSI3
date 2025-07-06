import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Análise de Qualidade de Vinhos")

@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('data/processed/wine-quality-combined.parquet')
        df['good_quality'] = (df['quality'] >= 6).astype(int)
        if 'type' in df.columns:
            df = df.rename(columns={'type': 'wine_type'})
        if df['wine_type'].dtype == 'object':
            df['wine_type'] = df['wine_type'].map({'red': 'Tinto', 'white': 'Branco'})
        df['qualidade_binaria'] = df['good_quality'].map({0: 'Ruim', 1: 'Bom'})
        return df
    except FileNotFoundError:
        st.error("Erro: O arquivo 'wine-quality-combined.parquet' não foi encontrado. Por favor, verifique o caminho.")
        st.stop()

df = load_data()

features_en = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]
features_pt = [
    'Acidez Fixa', 'Acidez Volátil', 'Ácido Cítrico', 'Açúcar Residual', 'Cloretos',
    'Dióxido de Enxofre Livre', 'Dióxido de Enxofre Total', 'Densidade', 'pH', 'Sulfatos', 'Teor Alcoólico'
]
feature_dict = dict(zip(features_en, features_pt))
feature_dict_inv = {v: k for k, v in feature_dict.items()}

wine_type_colors = {'Branco': '#f5f5f5', 'Tinto': '#800020'} 
qualidade_colors = {'Bom': '#2ecc40', 'Ruim': '#ff4136'}    

st.title("🍷 Análise Interativa da Qualidade de Vinhos")
st.markdown("""
Este projeto utiliza dados de vinhos 'Vinho Verde' (tinto e branco) para prever a qualidade dos vinhos com base em características físico-químicas. O objetivo é classificar vinhos em 'Bom' (nota >= 6) ou 'Ruim' (nota < 6), analisando padrões, correlações e variáveis importantes para a predição.
""")

st.sidebar.header("Opções de Filtragem")

wine_types = df['wine_type'].unique().tolist()
selected_wine_type = st.sidebar.multiselect(
    "Selecione o Tipo de Vinho:",
    options=wine_types,
    default=wine_types
)

min_quality, max_quality = int(df['quality'].min()), int(df['quality'].max())
quality_range = st.sidebar.slider(
    "Selecione a Faixa de Qualidade (0-10):",
    min_value=min_quality,
    max_value=max_quality,
    value=(min_quality, max_quality)
)

feature_filters = {}
for feat_en, feat_pt in zip(features_en, features_pt):
    min_val, max_val = float(df[feat_en].min()), float(df[feat_en].max())
    feature_filters[feat_en] = st.sidebar.slider(
        f"{feat_pt}",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=(0.01 if max_val-min_val < 10 else 0.1)
    )

df_filtered = df[
    (df['wine_type'].isin(selected_wine_type)) &
    (df['quality'] >= quality_range[0]) & (df['quality'] <= quality_range[1])
]
for feat_en in features_en:
    df_filtered = df_filtered[(df_filtered[feat_en] >= feature_filters[feat_en][0]) & (df_filtered[feat_en] <= feature_filters[feat_en][1])]

st.header("Visão Geral dos Dados Filtrados")
st.write(f"Total de amostras filtradas: **{len(df_filtered)}** de {len(df)}")
col1, col2, col3 = st.columns(3)
col1.metric("Vinhos Tintos", int((df_filtered['wine_type']=='Tinto').sum()))
col2.metric("Vinhos Brancos", int((df_filtered['wine_type']=='Branco').sum()))
col3.metric("Proporção de Bom", f"{100*df_filtered['good_quality'].mean():.1f}%")

st.subheader("Distribuição das Classes de Qualidade (Binária)")
class_counts = df_filtered['qualidade_binaria'].value_counts().reset_index()
class_counts.columns = ['Qualidade', 'Quantidade']
fig_class = px.bar(
    class_counts, x='Qualidade', y='Quantidade',
    title='Distribuição das Classes Binárias',
    labels={'Quantidade': 'Número de Vinhos', 'Qualidade': 'Qualidade'},
    color='Qualidade',
    color_discrete_map=qualidade_colors
)
st.plotly_chart(fig_class, use_container_width=True)

st.header("Análise Univariada")
st.markdown("Selecione uma variável físico-química para explorar sua distribuição.")
selected_uni_pt = st.selectbox("Escolha a variável:", features_pt, index=features_pt.index('Teor Alcoólico'))
selected_uni_en = feature_dict_inv[selected_uni_pt]
col_uni1, col_uni2 = st.columns(2)
with col_uni1:
    st.write(f"#### Histograma de {selected_uni_pt}")
    fig_uni = px.histogram(
        df_filtered, x=selected_uni_en, color='wine_type', barmode='overlay',
        nbins=30, opacity=0.7,
        labels={selected_uni_en: selected_uni_pt, 'wine_type': 'Tipo de Vinho'},
        color_discrete_map=wine_type_colors
    )
    st.plotly_chart(fig_uni, use_container_width=True)
with col_uni2:
    st.write(f"#### Boxplot de {selected_uni_pt} por Qualidade Binária")
    fig_box = px.box(
        df_filtered, x='qualidade_binaria', y=selected_uni_en, color='qualidade_binaria',
        labels={'qualidade_binaria': 'Qualidade', selected_uni_en: selected_uni_pt},
        color_discrete_map=qualidade_colors
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.header("Análise Bivariada")
st.markdown("Selecione duas variáveis para explorar relações e padrões.")
col_bi1, col_bi2 = st.columns(2)
with col_bi1:
    x_bi_pt = st.selectbox("Eixo X:", features_pt, index=features_pt.index('Teor Alcoólico'), key='x_bi')
    x_bi_en = feature_dict_inv[x_bi_pt]
with col_bi2:
    y_bi_pt = st.selectbox("Eixo Y:", features_pt, index=features_pt.index('Acidez Volátil'), key='y_bi')
    y_bi_en = feature_dict_inv[y_bi_pt]
fig_bi = px.scatter(
    df_filtered, x=x_bi_en, y=y_bi_en, color='qualidade_binaria', symbol='wine_type',
    labels={'qualidade_binaria': 'Qualidade', x_bi_en: x_bi_pt, y_bi_en: y_bi_pt, 'wine_type': 'Tipo de Vinho'},
    title=f'Relação entre {x_bi_pt} e {y_bi_pt}',
    color_discrete_map=qualidade_colors,
    symbol_map={'Branco': 'circle', 'Tinto': 'diamond'}
)
st.plotly_chart(fig_bi, use_container_width=True)

st.header("Matriz de Correlação das Variáveis Físico-Químicas")
corr = df_filtered[features_en + ['quality', 'good_quality']].corr()

corr_display = corr.copy()
corr_display.index = [feature_dict.get(c, c.capitalize()) if c in feature_dict else c.capitalize() for c in corr_display.index]
corr_display.columns = [feature_dict.get(c, c.capitalize()) if c in feature_dict else c.capitalize() for c in corr_display.columns]
fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_display, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

st.header("Visualização de Outliers")
st.markdown("Selecione uma variável para visualizar possíveis outliers por classe de qualidade.")
outlier_var_pt = st.selectbox("Variável para Outlier:", features_pt, index=features_pt.index('Açúcar Residual'))
outlier_var_en = feature_dict_inv[outlier_var_pt]
fig_out, ax_out = plt.subplots(figsize=(8, 4))
sns.boxplot(
    data=df_filtered, x='qualidade_binaria', y=outlier_var_en,
    palette=qualidade_colors
)
ax_out.set_xlabel('Qualidade')
ax_out.set_ylabel(outlier_var_pt)
st.pyplot(fig_out)

st.header("Importância das Variáveis para a Qualidade")
st.markdown("Correlação das variáveis físico-químicas com a qualidade (original e binária). Quanto mais próximo de 1 ou -1, mais forte a relação.")
corr_quality = corr['quality'].drop(['quality', 'good_quality']).sort_values(key=abs, ascending=False)
corr_good = corr['good_quality'].drop(['quality', 'good_quality']).sort_values(key=abs, ascending=False)
corr_quality.index = [feature_dict.get(c, c.capitalize()) for c in corr_quality.index]
corr_good.index = [feature_dict.get(c, c.capitalize()) for c in corr_good.index]
col_imp1, col_imp2 = st.columns(2)
with col_imp1:
    st.write("#### Correlação com Qualidade (0-10)")
    st.bar_chart(corr_quality)
with col_imp2:
    st.write("#### Correlação com Qualidade Binária")
    st.bar_chart(corr_good)