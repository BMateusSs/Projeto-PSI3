import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import StringIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análise de Vinhos Tintos", layout="wide")
st.title("Análise de Vinhos Tintos com K-Means")

st.markdown("""
Esta aplicação realiza uma análise de clustering em um conjunto de dados de vinhos tintos usando o algoritmo K-Means.
O objetivo é identificar grupos naturais de vinhos com características semelhantes e entender como esses grupos se relacionam com a qualidade do vinho.

**Como usar esta aplicação:**
1. Use o slider no menu lateral para ajustar o número de clusters
2. Explore os gráficos e análises gerados
3. Selecione clusters específicos para análises detalhadas
""")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/raw/winequality-red.csv')
        return df
    except FileNotFoundError:
        st.error("Arquivo 'winequality-red.csv' não encontrado. Por favor, verifique o caminho do arquivo.")
        return None

def standardize_data(df):
    features = df.drop('quality', axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, features.columns

def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

def apply_pca(data):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    return principal_components, pca

def plot_pca_clusters(pca_data, clusters, n_clusters):
    st.markdown("""
    ### Visualização por Componentes Principais (PCA)
    
    Este gráfico mostra a projeção dos vinhos em um espaço bidimensional usando PCA. 
    Cada ponto representa um vinho, e a cor indica a qual cluster ele pertence.
    
    **O que observar:**
    - Pontos da mesma cor representam vinhos agrupados no mesmo cluster
    - Clusters bem separados indicam que os grupos têm características distintas
    - Clusters sobrepostos podem indicar similaridades entre os grupos
    """)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    ax.set_title('PCA dos Clusters de Vinhos Tintos')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    st.pyplot(fig)

def plot_quality_by_cluster(df, clusters):
    st.markdown("""
    ### Distribuição de Qualidade por Cluster
    
    Este gráfico de caixas (boxplot) mostra como a qualidade dos vinhos varia dentro de cada cluster.
    
    **Como interpretar:**
    - A linha no meio da caixa representa a mediana (valor central)
    - A caixa representa os quartis (25% a 75% dos dados)
    - As linhas estendidas mostram o intervalo dos dados (excluindo outliers)
    - Pontos isolados são outliers (valores atípicos)
    
    **Insights possíveis:**
    - Clusters com medianas mais altas contêm vinhos de melhor qualidade
    - Caixas menores indicam clusters com qualidade mais consistente
    - Clusters com grande variação podem conter vinhos de qualidades diversas
    """)
    
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Cluster', y='quality', data=df_with_clusters, ax=ax)
    ax.set_title('Distribuição de Qualidade por Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Qualidade')
    st.pyplot(fig)

def plot_cluster_distribution(df, clusters, selected_cluster):
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == selected_cluster]
    
    st.write(f"### Estatísticas do Cluster {selected_cluster}")
    st.write(f"Número de vinhos neste cluster: {len(cluster_data)}")
    st.write(f"Qualidade média dos vinhos: {cluster_data['quality'].mean():.2f}")
    
    st.markdown("""
    ### Distribuição de Qualidade no Cluster Selecionado
    
    Este histograma mostra como as pontuações de qualidade estão distribuídas dentro do cluster escolhido.
    
    **O que observar:**
    - Picos no histograma indicam valores de qualidade mais comuns
    - A linha suave (KDE) mostra a tendência geral da distribuição
    - Distribuições concentradas em valores altos indicam clusters de vinhos premium
    - Distribuições concentradas em valores baixos indicam clusters de vinhos de qualidade inferior
    """)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(cluster_data['quality'], kde=True, ax=ax)
    ax.set_title(f'Distribuição de Qualidade no Cluster {selected_cluster}')
    ax.set_xlabel('Qualidade')
    ax.set_ylabel('Contagem')
    st.pyplot(fig)

def plot_quality_comparison(df, clusters, selected_cluster):
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    
    cluster_data = df_with_clusters[df_with_clusters['Cluster'] == selected_cluster]
    
    low_quality = cluster_data[cluster_data['quality'] <= 5]
    high_quality = cluster_data[cluster_data['quality'] >= 7]
    
    if low_quality.empty or high_quality.empty:
        st.warning(f"O cluster {selected_cluster} não contém ambos os tipos de qualidade (baixa <= 5 e alta >= 7).")
        return
    
    st.markdown("""
    ### Comparação de Propriedades Químicas por Qualidade
    
    Este gráfico compara as propriedades químicas médias entre vinhos de baixa qualidade (≤5) e alta qualidade (≥7) dentro do cluster selecionado.
    
    **O que procurar:**
    - Diferenças significativas entre as barras indicam fatores que podem influenciar a qualidade
    - Propriedades com valores mais altos em vinhos de alta qualidade podem ser características desejáveis
    - Propriedades com valores mais baixos em vinhos de alta qualidade podem ser características a evitar
    
    **Glossário de propriedades:**
    - **fixed acidity**: acidez fixa (principalmente ácido tartárico)
    - **volatile acidity**: acidez volátil (principalmente ácido acético)
    - **citric acid**: ácido cítrico
    - **residual sugar**: açúcar residual
    - **chlorides**: cloretos (sal)
    - **free sulfur dioxide**: dióxido de enxofre livre
    - **total sulfur dioxide**: dióxido de enxofre total
    - **density**: densidade
    - **pH**: nível de acidez/basicidade
    - **sulphates**: sulfatos (aditivo)
    - **alcohol**: teor alcoólico
    """)
    
    features = df.columns.drop('quality')
    low_means = low_quality[features].mean()
    high_means = high_quality[features].mean()
    
    comparison_df = pd.DataFrame({
        'Baixa Qualidade (<=5)': low_means,
        'Alta Qualidade (>=7)': high_means
    })
    
    fig, ax = plt.subplots(figsize=(14, 8))
    comparison_df.plot(kind='bar', ax=ax)
    ax.set_title(f'Comparação de Propriedades Químicas por Qualidade no Cluster {selected_cluster}')
    ax.set_ylabel('Valor Médio')
    ax.set_xlabel('Propriedades Químicas')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

def main():
    df = load_data()
    
    if df is not None:
        st.write("## Dataset de Vinhos Tintos")
        st.write(f"Dimensões: {df.shape[0]} linhas e {df.shape[1]} colunas")
        
        with st.expander("Sobre o Dataset"):
            st.markdown("""
            Este dataset contém informações sobre vinhos tintos, incluindo suas propriedades químicas e uma classificação de qualidade.
            Cada linha representa um vinho diferente, e cada coluna representa uma propriedade química ou a pontuação de qualidade.
            
            A qualidade é uma pontuação entre 0 e 10, com valores mais altos indicando vinhos de melhor qualidade.
            """)
        
        with st.expander("Visualizar Dados"):
            st.dataframe(df)
        
        with st.expander("Estatísticas Descritivas"):
            st.markdown("""
            A tabela abaixo mostra estatísticas resumidas para cada propriedade do vinho:
            - **count**: número de amostras
            - **mean**: valor médio
            - **std**: desvio padrão (medida de dispersão)
            - **min/max**: valores mínimo e máximo
            - **25%/50%/75%**: quartis (valores que dividem os dados em 4 partes)
            """)
            st.write(df.describe())
        
        scaled_features, feature_names = standardize_data(df)
        
        st.sidebar.header("Configurações")
        
        st.sidebar.markdown("""
        ### O que é o K-Means?
        
        K-Means é um algoritmo de clustering que agrupa dados similares.
        O parâmetro K define quantos grupos (clusters) serão formados.
        
        Use o slider abaixo para ajustar o número de clusters e observe como os padrões mudam nos gráficos.
        """)
        
        k_clusters = st.sidebar.slider("Número de Clusters (K)", min_value=2, max_value=10, value=10)
        
        clusters, kmeans_model = apply_kmeans(scaled_features, k_clusters)
        
        pca_data, pca_model = apply_pca(scaled_features)
        
        st.write(f"## Distribuição dos {k_clusters} Clusters")
        st.markdown("""
        Este gráfico de barras mostra quantos vinhos foram classificados em cada cluster.
        
        **O que observar:**
        - Clusters com mais vinhos são mais comuns no dataset
        - Clusters pequenos podem representar vinhos com características raras ou únicas
        """)
        
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        st.bar_chart(cluster_counts)
        
        dist_selected_cluster = st.selectbox("Selecione um cluster para análise de distribuição", 
                                         range(k_clusters), 
                                         format_func=lambda x: f"Cluster {x}")
        
        plot_cluster_distribution(df, clusters, dist_selected_cluster)
        
        st.write("## Visualização PCA dos Clusters")
        plot_pca_clusters(pca_data, clusters, k_clusters)
        
        st.write("## Distribuição de Qualidade por Cluster")
        plot_quality_by_cluster(df, clusters)
        
        selected_cluster = st.selectbox("Selecione um cluster para análise detalhada", 
                                        range(k_clusters), 
                                        format_func=lambda x: f"Cluster {x}")
        
        st.write(f"## Comparação de Vinhos de Baixa vs. Alta Qualidade no Cluster {selected_cluster}")
        plot_quality_comparison(df, clusters, selected_cluster)
        st.write("## Centroides dos Clusters")
        st.markdown("""
        A tabela abaixo mostra os centroides (pontos centrais) de cada cluster.
        
        **O que são centroides?**
        Os centroides representam o "perfil médio" de cada cluster. Cada valor representa a média dessa propriedade para os vinhos daquele cluster.
        
        **Como interpretar:**
        - Compare os valores entre clusters para entender o que torna cada grupo único
        - Valores mais altos ou mais baixos podem indicar características distintivas de cada grupo
        """)
        
        centroids_df = pd.DataFrame(kmeans_model.cluster_centers_, columns=feature_names)
        centroids_df.index.name = 'Cluster'
        st.dataframe(centroids_df)
        
        st.markdown("""
        ## Conclusões e Recomendações
        
        Com base nesta análise de clustering:
        
        1. Os vinhos podem ser agrupados com base em suas propriedades químicas, formando clusters distintos
        2. Alguns clusters contêm vinhos de qualidade consistentemente mais alta que outros
        3. Ao comparar vinhos de alta e baixa qualidade dentro do mesmo cluster, podemos identificar quais propriedades químicas têm maior impacto na qualidade
        
        **Possíveis aplicações:**
        - Produtores de vinho podem otimizar suas técnicas para favorecer as propriedades associadas a vinhos de alta qualidade
        - Especialistas em vinho podem usar os clusters para recomendar vinhos similares
        - Consumidores podem entender melhor as características que definem seus vinhos favoritos
        """)

if __name__ == "__main__":
    main()