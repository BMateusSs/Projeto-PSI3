import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

def aplicar_dbscan_e_kmeans_no_cluster_gigante():
    """
    Demonstra como aplicar K-means apenas no cluster gigante identificado pelo DBSCAN.
    """
    
    print("=== Aplicando K-means no Cluster Gigante do DBSCAN ===\n")
    
    # 1. Carregar os dados (assumindo que você tem embeddings 2D)
    try:
        # Carregar embeddings 2D (ajuste o caminho conforme necessário)
        embeddings_2d = np.load('embeddings_2d.npy')
        print(f"Dados carregados: {embeddings_2d.shape}")
    except FileNotFoundError:
        print("Arquivo embeddings_2d.npy não encontrado. Criando dados de exemplo...")
        # Criar dados de exemplo para demonstração
        np.random.seed(42)
        n_samples = 10000
        
        # Simular um cluster gigante (80% dos dados)
        cluster_gigante = np.random.normal(0, 1, (int(n_samples * 0.8), 2))
        
        # Simular alguns clusters menores
        cluster_pequeno1 = np.random.normal(5, 0.5, (int(n_samples * 0.1), 2))
        cluster_pequeno2 = np.random.normal(-5, 0.5, (int(n_samples * 0.05), 2))
        cluster_pequeno3 = np.random.normal([5, -5], 0.3, (int(n_samples * 0.05), 2))
        
        embeddings_2d = np.vstack([cluster_gigante, cluster_pequeno1, cluster_pequeno2, cluster_pequeno3])
        print(f"Dados de exemplo criados: {embeddings_2d.shape}")
    
    # 2. Aplicar DBSCAN para identificar clusters
    print("\n1. Aplicando DBSCAN para identificar clusters...")
    
    # Parâmetros do DBSCAN (ajuste conforme necessário)
    eps = 0.255
    min_samples = 40
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters_dbscan = dbscan.fit_predict(embeddings_2d)
    
    # Analisar resultados do DBSCAN
    n_clusters = len(set(clusters_dbscan)) - (1 if -1 in clusters_dbscan else 0)
    n_noise = list(clusters_dbscan).count(-1)
    
    print(f"DBSCAN encontrou {n_clusters} clusters")
    print(f"Número de pontos de ruído: {n_noise}")
    
    # Contar pontos em cada cluster
    unique_clusters, counts = np.unique(clusters_dbscan, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters, counts))
    
    print("\nTamanho de cada cluster:")
    for cluster_id, size in cluster_sizes.items():
        if cluster_id == -1:
            print(f"  Ruído: {size} pontos")
        else:
            print(f"  Cluster {cluster_id}: {size} pontos")
    
    # 3. Identificar o cluster gigante
    print("\n2. Identificando o cluster gigante...")
    
    # Encontrar o cluster com mais pontos (excluindo ruído)
    cluster_sizes_no_noise = {k: v for k, v in cluster_sizes.items() if k != -1}
    cluster_gigante_id = max(cluster_sizes_no_noise, key=cluster_sizes_no_noise.get)
    tamanho_cluster_gigante = cluster_sizes_no_noise[cluster_gigante_id]
    
    print(f"Cluster gigante identificado: Cluster {cluster_gigante_id}")
    print(f"Tamanho do cluster gigante: {tamanho_cluster_gigante} pontos")
    print(f"Percentual do total: {tamanho_cluster_gigante/len(embeddings_2d)*100:.1f}%")
    
    # 4. Filtrar dados do cluster gigante
    print("\n3. Filtrando dados do cluster gigante...")
    
    # Criar máscara para o cluster gigante
    mask_cluster_gigante = clusters_dbscan == cluster_gigante_id
    dados_cluster_gigante = embeddings_2d[mask_cluster_gigante]
    
    print(f"Dados filtrados do cluster gigante: {dados_cluster_gigante.shape}")
    
    # 5. Aplicar K-means apenas no cluster gigante
    print("\n4. Aplicando K-means no cluster gigante...")
    
    # Definir número de clusters para K-means (ajuste conforme necessário)
    n_clusters_kmeans = 5
    
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
    clusters_kmeans = kmeans.fit_predict(dados_cluster_gigante)
    
    print(f"K-means aplicado com {n_clusters_kmeans} clusters")
    
    # 6. Analisar resultados do K-means
    print("\n5. Analisando resultados do K-means...")
    
    unique_kmeans_clusters, kmeans_counts = np.unique(clusters_kmeans, return_counts=True)
    kmeans_cluster_sizes = dict(zip(unique_kmeans_clusters, kmeans_counts))
    
    print("Tamanho dos sub-clusters K-means:")
    for cluster_id, size in kmeans_cluster_sizes.items():
        print(f"  Sub-cluster K-means {cluster_id}: {size} pontos")
    
    # 7. Visualizar resultados
    print("\n6. Criando visualizações...")
    
    # Criar figura com subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Resultado original do DBSCAN
    ax1 = axes[0]
    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=clusters_dbscan, cmap='viridis', alpha=0.6, s=1)
    ax1.set_title(f'DBSCAN - {n_clusters} clusters identificados')
    ax1.set_xlabel('Dimensão 1')
    ax1.set_ylabel('Dimensão 2')
    plt.colorbar(scatter1, ax=ax1, label='Cluster DBSCAN')
    
    # Plot 2: Apenas o cluster gigante destacado
    ax2 = axes[1]
    # Plotar todos os pontos em cinza
    ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
               c='lightgray', alpha=0.3, s=1, label='Outros clusters')
    # Destacar o cluster gigante
    ax2.scatter(dados_cluster_gigante[:, 0], dados_cluster_gigante[:, 1], 
               c='red', alpha=0.7, s=2, label=f'Cluster gigante ({cluster_gigante_id})')
    ax2.set_title(f'Cluster Gigante Destacado\n({tamanho_cluster_gigante} pontos)')
    ax2.set_xlabel('Dimensão 1')
    ax2.set_ylabel('Dimensão 2')
    ax2.legend()
    
    # Plot 3: K-means aplicado no cluster gigante
    ax3 = axes[2]
    scatter3 = ax3.scatter(dados_cluster_gigante[:, 0], dados_cluster_gigante[:, 1], 
                          c=clusters_kmeans, cmap='plasma', alpha=0.7, s=2)
    # Plotar centroides do K-means
    centroids = kmeans.cluster_centers_
    ax3.scatter(centroids[:, 0], centroids[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroides K-means')
    ax3.set_title(f'K-means no Cluster Gigante\n({n_clusters_kmeans} sub-clusters)')
    ax3.set_xlabel('Dimensão 1')
    ax3.set_ylabel('Dimensão 2')
    plt.colorbar(scatter3, ax=ax3, label='Sub-cluster K-means')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('dbscan_kmeans_cluster_gigante.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Criar DataFrame com resultados
    print("\n7. Criando DataFrame com resultados...")
    
    # Criar DataFrame original com clusters DBSCAN
    df_original = pd.DataFrame(embeddings_2d, columns=['dim1', 'dim2'])
    df_original['cluster_dbscan'] = clusters_dbscan
    
    # Criar DataFrame do cluster gigante com K-means
    df_cluster_gigante = pd.DataFrame(dados_cluster_gigante, columns=['dim1', 'dim2'])
    df_cluster_gigante['sub_cluster_kmeans'] = clusters_kmeans
    df_cluster_gigante['cluster_dbscan_original'] = cluster_gigante_id
    
    print("\nResumo dos resultados:")
    print(f"- Total de pontos: {len(embeddings_2d)}")
    print(f"- Clusters DBSCAN identificados: {n_clusters}")
    print(f"- Cluster gigante: {cluster_gigante_id} ({tamanho_cluster_gigante} pontos)")
    print(f"- Sub-clusters K-means no cluster gigante: {n_clusters_kmeans}")
    
    # Salvar resultados
    df_original.to_csv('resultados_dbscan_completo.csv', index=False)
    df_cluster_gigante.to_csv('resultados_kmeans_cluster_gigante.csv', index=False)
    
    print("\nArquivos salvos:")
    print("- resultados_dbscan_completo.csv")
    print("- resultados_kmeans_cluster_gigante.csv")
    print("- dbscan_kmeans_cluster_gigante.png")
    
    return df_original, df_cluster_gigante, kmeans

def analisar_sub_clusters_kmeans(df_cluster_gigante, kmeans_model):
    """
    Analisa os sub-clusters criados pelo K-means no cluster gigante.
    """
    print("\n=== Análise Detalhada dos Sub-clusters K-means ===\n")
    
    # Estatísticas por sub-cluster
    print("Estatísticas por sub-cluster:")
    stats_por_cluster = df_cluster_gigante.groupby('sub_cluster_kmeans').agg({
        'dim1': ['count', 'mean', 'std'],
        'dim2': ['mean', 'std']
    }).round(3)
    
    print(stats_por_cluster)
    
    # Análise de silhueta (se disponível)
    try:
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(df_cluster_gigante[['dim1', 'dim2']], 
                                        df_cluster_gigante['sub_cluster_kmeans'])
        print(f"\nCoeficiente de Silhueta: {silhouette_avg:.3f}")
        
        if silhouette_avg > 0.7:
            print("✓ Boa separação entre clusters")
        elif silhouette_avg > 0.5:
            print("✓ Separação moderada entre clusters")
        else:
            print("⚠ Separação fraca entre clusters")
            
    except ImportError:
        print("\nMétrica de silhueta não disponível (sklearn.metrics.silhouette_score)")
    
    # Visualizar distribuição dos sub-clusters
    plt.figure(figsize=(12, 5))
    
    # Histograma de distribuição
    plt.subplot(1, 2, 1)
    df_cluster_gigante['sub_cluster_kmeans'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribuição dos Sub-clusters K-means')
    plt.xlabel('Sub-cluster')
    plt.ylabel('Número de Pontos')
    plt.xticks(rotation=0)
    
    # Boxplot das dimensões por cluster
    plt.subplot(1, 2, 2)
    df_cluster_gigante.boxplot(column='dim1', by='sub_cluster_kmeans', ax=plt.gca())
    plt.title('Distribuição da Dimensão 1 por Sub-cluster')
    plt.suptitle('')  # Remove título automático
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Executar análise completa
    df_original, df_cluster_gigante, kmeans_model = aplicar_dbscan_e_kmeans_no_cluster_gigante()
    
    # Análise detalhada dos sub-clusters
    analisar_sub_clusters_kmeans(df_cluster_gigante, kmeans_model)
    
    print("\n=== Análise Concluída ===")
    print("Este script demonstra como:")
    print("1. Aplicar DBSCAN para identificar clusters")
    print("2. Identificar o cluster gigante")
    print("3. Aplicar K-means apenas no cluster gigante")
    print("4. Analisar os sub-clusters resultantes") 