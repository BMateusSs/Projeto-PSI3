import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análise de Vinhos por Clustering", layout="wide")
st.title("Análise de Qualidade de Vinhos com K-Means")

st.markdown("""
### Sobre esta análise
Esta aplicação analisa dados de qualidade de vinhos usando o algoritmo K-Means, que agrupa vinhos com características similares.
Os clusters formados podem revelar padrões interessantes sobre como as propriedades químicas se relacionam com a qualidade do vinho.
""")

@st.cache_data
def load_data():
    df = pd.read_csv('data/raw/combined_wine_quality.csv', sep=';')
    categorical_cols = []
    for col in df.columns:
        if col == 'quality':
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            categorical_cols.append(col)
    
    categorical_data = {col: df[col] for col in categorical_cols}
    
    df = df.drop(columns=categorical_cols)
    
    return df, categorical_data

try:
    df, categorical_data = load_data()
    
    st.subheader("Primeiras linhas do dataset")
    st.dataframe(df.head())
    
    st.markdown("""
    #### Compreendendo os dados
    Os dados acima mostram as características químicas de diferentes vinhos, como acidez, níveis de açúcar, pH, 
    e outros compostos, junto com uma classificação de qualidade atribuída por especialistas.
    """)
    
    st.subheader("Informações do Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Número de amostras: {df.shape[0]}")
        st.write(f"Número de características: {df.shape[1]}")
    with col2:
        st.write("Estatísticas descritivas:")
        st.write(df.describe())
    
    X = df.drop('quality', axis=1) if 'quality' in df.columns else df
    y = df['quality'] if 'quality' in df.columns else None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.subheader("K-Means Clustering")
    
    st.markdown("""
    #### O que é K-Means?
    K-Means é um algoritmo de agrupamento que divide os dados em K grupos distintos. 
    Cada grupo contém amostras (vinhos) com características similares. 
    Use o slider abaixo para escolher quantos grupos você deseja formar.
    """)
    
    k = st.slider("Selecione o número de clusters (k)", min_value=2, max_value=10, value=3, 
                 help="Um número maior de clusters cria grupos menores e mais específicos")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(
        data=X_pca, 
        columns=['PC1', 'PC2']
    )
    pca_df['cluster'] = clusters
    if y is not None:
        pca_df['quality'] = y.values
    
    st.subheader("Visualização dos Clusters (PCA)")
    
    st.markdown("""
    #### Entendendo o gráfico PCA
    Este gráfico mostra os vinhos agrupados em clusters, reduzidos a duas dimensões para facilitar a visualização.
    - Cada ponto representa um vinho
    - Cores diferentes representam clusters diferentes
    - Pontos próximos têm características químicas semelhantes
    - A distância entre clusters indica quão diferentes são seus vinhos
    """)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(k):
        cluster_data = pca_df[pca_df['cluster'] == i]
        ax.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {i}')
    
    ax.set_title('Clusters visualizados com PCA')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.legend()
    st.pyplot(fig)
    
    explained_variance = pca.explained_variance_ratio_
    st.write(f"Variância explicada pela PC1: {explained_variance[0]:.2f}")
    st.write(f"Variância explicada pela PC2: {explained_variance[1]:.2f}")
    st.write(f"Variância total explicada: {sum(explained_variance):.2f}")
    
    st.info("""
    **O que significa variância explicada?** 
    É a quantidade de informação original preservada após a redução dimensional. 
    Quanto mais próximo de 1.0 (ou 100%), melhor a visualização representa os dados originais.
    """)
    
    if y is not None:
        st.subheader("Distribuição de Qualidade por Cluster")
        
        st.markdown("""
        #### Como interpretar este boxplot
        Este gráfico mostra a distribuição das pontuações de qualidade em cada cluster:
        - A linha no meio de cada caixa é a mediana (valor central)
        - As bordas da caixa mostram o 1º e 3º quartis (25% e 75% dos dados)
        - As "antenas" mostram os valores mínimos e máximos (excluindo outliers)
        - Pontos isolados são outliers (valores muito diferentes do restante do grupo)
        
        Um cluster com caixas mais altas contém vinhos de melhor qualidade em média.
        """)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='cluster', y='quality', data=df_with_clusters, ax=ax)
        ax.set_title('Distribuição de Qualidade por Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Qualidade')
        st.pyplot(fig)
    
    st.subheader("Análise Detalhada por Cluster")
    st.markdown("""
    Selecione um cluster específico para analisar em detalhes suas características e diferenças 
    entre vinhos de alta e baixa qualidade dentro desse grupo.
    """)
    
    selected_cluster = st.selectbox("Selecione um cluster para análise detalhada", 
                                    options=list(range(k)))
    
    cluster_data = df_with_clusters[df_with_clusters['cluster'] == selected_cluster]
    
    if y is not None:
        low_quality = cluster_data[cluster_data['quality'] <= 5]
        high_quality = cluster_data[cluster_data['quality'] >= 7]
        
        st.write(f"Cluster {selected_cluster}:")
        st.write(f"Total de vinhos: {len(cluster_data)}")
        st.write(f"Vinhos de baixa qualidade (<=5): {len(low_quality)}")
        st.write(f"Vinhos de alta qualidade (>=7): {len(high_quality)}")
        
        st.subheader(f"Comparação de Propriedades Químicas no Cluster {selected_cluster}")
        
        if len(low_quality) > 0 and len(high_quality) > 0:
            st.markdown("""
            #### Como interpretar o gráfico de comparação
            Este gráfico compara as propriedades químicas médias entre vinhos de baixa e alta qualidade:
            - Barras mais altas indicam valores médios maiores para aquela propriedade
            - Diferenças significativas entre barras azuis e laranjas revelam características 
              que podem influenciar a qualidade do vinho
            - Atenção especial para propriedades com grandes diferenças entre os grupos
            """)
            
            low_means = low_quality.drop(['quality', 'cluster'], axis=1).mean()
            high_means = high_quality.drop(['quality', 'cluster'], axis=1).mean()
            
            comparison_df = pd.DataFrame({
                'Baixa Qualidade (<=5)': low_means,
                'Alta Qualidade (>=7)': high_means
            })
            
            fig, ax = plt.subplots(figsize=(14, 8))
            comparison_df.plot(kind='bar', ax=ax)
            ax.set_title(f'Propriedades Químicas: Baixa vs Alta Qualidade no Cluster {selected_cluster}')
            ax.set_ylabel('Valor Médio')
            ax.set_xlabel('Propriedade Química')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.write("Tabela de Comparação:")
            st.dataframe(comparison_df)
            
            comparison_df['Diferença (%)'] = ((high_means - low_means) / low_means * 100).round(2)
            top_diff = comparison_df['Diferença (%)'].abs().sort_values(ascending=False)
            
            st.markdown("#### Principais diferenças entre vinhos de alta e baixa qualidade")
            st.markdown(f"""
            As propriedades químicas com maior diferença percentual entre vinhos de alta e baixa qualidade são:
            1. **{top_diff.index[0]}**: {top_diff.iloc[0]}%
            2. **{top_diff.index[1]}**: {top_diff.iloc[1]}%
            3. **{top_diff.index[2]}**: {top_diff.iloc[2]}%
            
            Estas características podem ser mais importantes para determinar a qualidade do vinho neste cluster.
            """)
            
        else:
            st.warning(f"Não há dados suficientes para comparar vinhos de baixa e alta qualidade no Cluster {selected_cluster}.")
    
    st.subheader(f"Estatísticas do Cluster {selected_cluster}")
    st.dataframe(cluster_data.describe())
    
    st.markdown("""
    #### Interpretando as estatísticas
    A tabela acima mostra um resumo estatístico das propriedades químicas dos vinhos deste cluster:
    - **count**: número de amostras
    - **mean**: valor médio
    - **std**: desvio padrão (indica variabilidade)
    - **min/max**: valores mínimos e máximos
    - **25%/50%/75%**: percentis (o valor de 50% é a mediana)
    """)

except Exception as e:
    st.error(f"Erro ao carregar ou processar os dados: {e}")
    st.info("Verifique se o arquivo 'combined_wine_quality.csv' está no diretório correto e se o separador é ';'.")
