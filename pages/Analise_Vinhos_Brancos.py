import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Análise de Propriedades dos Vinhos Brancos", page_icon="🍷", layout="wide")

st.title("Análise das Propriedades dos Vinhos Brancos e sua Relação com a Qualidade")

@st.cache_data
def load_data():
    data = pd.read_csv("data/raw/winequality-white.csv", sep=";")
    return data

try:
    df = load_data()    
    st.subheader("Visão Geral dos Dados")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Número de amostras:** {df.shape[0]}")
    with col2:
        st.write(f"**Número de características:** {df.shape[1] - 1}")  
    st.subheader("Primeiras Linhas do Dataset")
    st.dataframe(df.head())
    
    st.subheader("Estatísticas Descritivas")
    st.dataframe(df.describe().style.format("{:.2f}"))
    
    st.subheader("Distribuição da Qualidade dos Vinhos Brancos")
    fig, ax = plt.subplots(figsize=(10, 6))
    quality_counts = df['quality'].value_counts().sort_index()
    
    colors = sns.color_palette("YlGnBu", len(quality_counts))
    
    bars = ax.bar(quality_counts.index, quality_counts.values, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Distribuição da Qualidade dos Vinhos Brancos', fontsize=14)
    ax.set_xlabel('Qualidade (escala de 0-10)', fontsize=12)
    ax.set_ylabel('Contagem de Vinhos', fontsize=12)
    ax.set_xticks(quality_counts.index)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    avg_quality = df['quality'].mean()
    ax.axvline(x=avg_quality, color='red', linestyle='--', alpha=0.8)
    ax.text(avg_quality + 0.1, max(quality_counts.values) * 0.9, 
            f'Média: {avg_quality:.2f}', color='red', fontsize=10)
    
    st.pyplot(fig)
    
    st.subheader("Correlação entre Propriedades e Qualidade")
    
    st.info("""
        **Como interpretar o mapa de calor:**
        - Valores próximos a 1 (azul escuro) indicam forte correlação positiva
        - Valores próximos a -1 (vermelho escuro) indicam forte correlação negativa
        - Valores próximos a 0 (branco) indicam pouca ou nenhuma correlação
    """)
    
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f', ax=ax, annot_kws={"size": 9})
    
    for i, label in enumerate(corr.columns):
        if label == 'quality':
            heatmap.get_xticklabels()[i].set_color('darkgreen')
            heatmap.get_yticklabels()[i].set_color('darkgreen')
            heatmap.get_xticklabels()[i].set_fontweight('bold')
            heatmap.get_yticklabels()[i].set_fontweight('bold')
    
    plt.title('Correlação entre as Propriedades dos Vinhos', fontsize=14, pad=20)
    st.pyplot(fig)
    
    quality_corr = corr['quality'].drop('quality').sort_values(ascending=False)
    
    st.subheader("Propriedades Mais Correlacionadas com a Qualidade")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Correlações Positivas:**")
        st.write(quality_corr[quality_corr > 0].to_frame().style.background_gradient(cmap='Blues').format("{:.3f}"))
    with col2:
        st.write("**Correlações Negativas:**")
        st.write(quality_corr[quality_corr < 0].to_frame().style.background_gradient(cmap='Reds_r').format("{:.3f}"))
    
    st.subheader("Explorar Relação entre Características e Qualidade")
    features = df.columns.tolist()
    features.remove('quality')
    
    selected_feature = st.selectbox("Escolha uma característica para visualizar sua relação com a qualidade:", 
                                   features, index=features.index(quality_corr.idxmax()) if not quality_corr.empty else 0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    palette = sns.color_palette("viridis", len(df['quality'].unique()))
    
    sns.boxplot(x='quality', y=selected_feature, data=df, ax=ax, palette=palette, showfliers=False)
    sns.stripplot(x='quality', y=selected_feature, data=df, ax=ax, color='black', alpha=0.3, size=3, jitter=True)
    
    for i, quality in enumerate(sorted(df['quality'].unique())):
        subset = df[df['quality'] == quality][selected_feature]
        ax.hlines(y=subset.median(), xmin=i-0.3, xmax=i+0.3, color='red', linestyle='-', linewidth=2, alpha=0.7)
    
    ax.set_title(f'Relação entre {selected_feature} e Qualidade', fontsize=14)
    ax.set_xlabel('Qualidade', fontsize=12)
    ax.set_ylabel(selected_feature, fontsize=12)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Mediana'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=6, alpha=0.3, label='Amostras individuais')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    st.subheader(f"Gráfico de Dispersão: {selected_feature} vs Qualidade")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    quality_jittered = df['quality'] + np.random.normal(0, 0.1, size=len(df))
    
    scatter = sns.scatterplot(x=selected_feature, y=quality_jittered, data=df, 
                             hue='quality', palette='viridis', alpha=0.6, s=50, ax=ax)
    
    sns.regplot(x=selected_feature, y='quality', data=df, scatter=False, 
               line_kws={"color": "red", "lw": 2, "linestyle": "--"}, ax=ax)
    
    corr_value = df[[selected_feature, 'quality']].corr().iloc[0,1]
    ax.text(0.05, 0.95, f'Correlação: {corr_value:.3f}', 
           transform=ax.transAxes, fontsize=12, 
           bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_title(f'Dispersão de {selected_feature} vs Qualidade', fontsize=14)
    ax.set_xlabel(selected_feature, fontsize=12)
    ax.set_ylabel('Qualidade (com jitter para melhor visualização)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    ax.legend_.remove()
    
    st.pyplot(fig)
    
    st.subheader("Comparação entre Duas Características")
    
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Escolha a primeira característica:", features, index=0)
    with col2:
        remaining_features = [f for f in features if f != feature1]
        feature2 = st.selectbox("Escolha a segunda característica:", remaining_features, 
                              index=0 if len(remaining_features) > 0 else 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = sns.scatterplot(x=feature1, y=feature2, hue='quality', size='quality',
                            palette='viridis', data=df, alpha=0.7, ax=ax)
    
    corr_feat = df[[feature1, feature2]].corr().iloc[0,1]
    ax.text(0.05, 0.95, f'Correlação entre {feature1} e {feature2}: {corr_feat:.3f}', 
           transform=ax.transAxes, fontsize=10, 
           bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_title(f'Relação entre {feature1}, {feature2} e Qualidade', fontsize=14)
    ax.set_xlabel(feature1, fontsize=12)
    ax.set_ylabel(feature2, fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    quality_label_indices = [i for i, label in enumerate(labels) if label.isdigit()]
    handles = [handles[i] for i in quality_label_indices[:len(df['quality'].unique())]]
    labels = [labels[i] for i in quality_label_indices[:len(df['quality'].unique())]]
    ax.legend(handles, labels, title='Qualidade', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"Erro ao carregar o dataset: {e}")
    st.info("Verifique se o arquivo 'winequality-white.csv' existe no diretório do aplicativo.")