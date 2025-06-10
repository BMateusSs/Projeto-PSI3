import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Análise de Vinhos Brancos",
    page_icon="🍷",
    layout="wide"
)

st.title("Análise de Vinhos Brancos por Clusters")

st.markdown("""
Esta aplicação analisa dados de vinhos brancos portugueses e utiliza algoritmos de clusterização para agrupar 
vinhos com características químicas similares. Isto nos ajuda a entender quais propriedades químicas 
influenciam a qualidade do vinho.
""")

@st.cache_data
def load_data():
    df = pd.read_csv('data/raw/winequality-white.csv', sep=';')
    return df

try:
    df = load_data()
    
    st.subheader("Primeiras linhas do dataset")
    st.markdown("""
    Abaixo estão as primeiras linhas do conjunto de dados, mostrando as características químicas 
    e pontuações de qualidade dos vinhos brancos.
    """)
    st.dataframe(df.head())
    
    st.subheader("Informações do dataset")
    st.markdown("""
    Este dataset contém diversas propriedades químicas dos vinhos brancos, como acidez, níveis de açúcar, 
    e álcool, junto com uma pontuação de qualidade (de 0 a 10) atribuída por especialistas.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Total de registros: {df.shape[0]}")
        st.write(f"Total de colunas: {df.shape[1]}")
    with col2:
        st.write("Estatísticas descritivas:")
        st.dataframe(df.describe())
    
    features = df.drop('quality', axis=1)
    feature_names = features.columns.tolist()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    st.subheader("Clusterização K-Means")
    st.markdown("""
    A clusterização K-Means agrupa os vinhos com base em suas propriedades químicas. 
    Mova o slider abaixo para alterar o número de grupos (clusters) e observe como os vinhos são reorganizados.
    
    **O que significa:** Vinhos no mesmo cluster têm propriedades químicas semelhantes, independentemente de sua qualidade.
    """)
    k = st.slider("Selecione o número de clusters (k)", min_value=2, max_value=10, value=3)
    
    @st.cache_data
    def apply_kmeans(scaled_data, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        return clusters, kmeans
    
    clusters, kmeans_model = apply_kmeans(scaled_features, k)
    
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    
    st.subheader("Visualização dos Clusters (PCA)")
    st.markdown("""
    Este gráfico mostra como os vinhos se agrupam quando reduzimos todas as características a apenas duas dimensões.
    
    **Como interpretar:** 
    - Cada ponto representa um vinho
    - Cores diferentes representam clusters diferentes
    - Pontos próximos têm características químicas similares
    - Agrupamentos bem definidos (com cores separadas) indicam clusters bem formados
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        principal_components[:, 0], 
        principal_components[:, 1], 
        c=clusters, 
        cmap='viridis', 
        alpha=0.6
    )
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.set_title('Clusters de Vinhos Brancos (PCA)')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)
    
    st.subheader("Qualidade por Cluster")
    st.markdown("""
    Este gráfico mostra como as pontuações de qualidade estão distribuídas em cada cluster.
    
    **Como interpretar:**
    - A linha horizontal dentro de cada caixa representa a mediana da qualidade
    - As bordas das caixas mostram o primeiro e terceiro quartis (25% e 75%)
    - As linhas verticais (whiskers) mostram a faixa de variação, excluindo outliers
    - Pontos individuais são outliers (valores muito distantes da média)
    
    **O que procurar:** Clusters com medianas mais altas tendem a conter vinhos de melhor qualidade.
    """)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='cluster', y='quality', data=df_with_clusters, ax=ax)
    ax.set_title('Distribuição da Qualidade por Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Qualidade')
    st.pyplot(fig)
    
    st.subheader("Análise Detalhada por Cluster")
    st.markdown("""
    Selecione um cluster específico para analisar detalhadamente sua composição e características.
    """)
    selected_cluster = st.selectbox("Selecione um cluster para análise", sorted(df_with_clusters['cluster'].unique()))
    
    cluster_data = df_with_clusters[df_with_clusters['cluster'] == selected_cluster]
    
    cluster_data['quality_category'] = pd.cut(
        cluster_data['quality'], 
        bins=[0, 5, 6, 10], 
        labels=['Baixa (<=5)', 'Média (6)', 'Alta (>=7)']
    )
    
    st.write(f"Distribuição da qualidade no cluster {selected_cluster}:")
    st.markdown("""
    Esta análise mostra como os vinhos deste cluster se distribuem entre as categorias de qualidade.
    """)
    quality_counts = cluster_data['quality_category'].value_counts().reset_index()
    quality_counts.columns = ['Categoria', 'Contagem']
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(quality_counts)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='quality_category', data=cluster_data, ax=ax)
        ax.set_title(f'Distribuição de Qualidade no Cluster {selected_cluster}')
        st.pyplot(fig)
    
    st.subheader(f"Comparação de Propriedades Químicas por Qualidade no Cluster {selected_cluster}")
    st.markdown("""
    Este gráfico compara as propriedades químicas médias entre vinhos de baixa qualidade (≤5) e alta qualidade (≥7) 
    dentro do cluster selecionado.
    
    **Como interpretar:**
    - Barras mais altas indicam valores médios maiores para aquela propriedade
    - Diferenças significativas entre as barras azuis e laranjas para uma mesma propriedade 
      sugerem que essa característica pode ser importante para a qualidade do vinho
    
    **Insight:** Procure as maiores diferenças entre vinhos de baixa e alta qualidade para identificar 
    quais propriedades químicas mais influenciam a qualidade.
    """)
    
    low_quality = cluster_data[cluster_data['quality'] <= 5]
    high_quality = cluster_data[cluster_data['quality'] >= 7]
    
    if len(low_quality) > 0 and len(high_quality) > 0:
        comparison_data = pd.DataFrame({
            'Propriedade': feature_names,
            'Baixa Qualidade (<=5)': low_quality[feature_names].mean().values,
            'Alta Qualidade (>=7)': high_quality[feature_names].mean().values
        })
        
        st.dataframe(comparison_data)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        comparison_data_melted = pd.melt(
            comparison_data, 
            id_vars=['Propriedade'], 
            var_name='Categoria de Qualidade', 
            value_name='Valor Médio'
        )
        
        sns.barplot(
            x='Propriedade', 
            y='Valor Médio', 
            hue='Categoria de Qualidade', 
            data=comparison_data_melted,
            ax=ax
        )
        
        plt.xticks(rotation=45)
        plt.title(f'Comparação de Propriedades Químicas por Qualidade no Cluster {selected_cluster}')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Não há dados suficientes de vinhos de baixa ou alta qualidade neste cluster para fazer a comparação.")
        st.markdown("""
        Este cluster não possui exemplos suficientes em ambas as categorias (baixa e alta qualidade) 
        para permitir uma comparação significativa. Isso pode indicar que este cluster é mais homogêneo 
        em termos de qualidade.
        """)
    
    st.subheader("Conclusões")
    st.markdown("""
    Esta análise permite identificar:
    
    1. Como os vinhos se agrupam naturalmente com base em suas propriedades químicas
    2. Quais clusters tendem a conter vinhos de maior qualidade
    3. Quais propriedades químicas diferenciam vinhos de alta e baixa qualidade dentro de um mesmo cluster
    
    Essas informações podem ajudar produtores a entender quais aspectos químicos mais contribuem para a 
    qualidade percebida dos vinhos brancos.
    """)
    
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.info("Verifique se o arquivo 'winequality-white.csv' está disponível no diretório correto.")
