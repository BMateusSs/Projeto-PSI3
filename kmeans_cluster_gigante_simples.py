"""
Script simples para aplicar K-means apenas no cluster gigante identificado pelo DBSCAN.

Este script assume que você já tem:
1. Seus embeddings 2D (embeddings_2d)
2. Os resultados do DBSCAN (clusters_dbscan)

Use este código diretamente no seu notebook ou script.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def aplicar_kmeans_no_cluster_gigante(embeddings_2d, clusters_dbscan, n_clusters_kmeans=5):
    """
    Aplica K-means apenas no cluster gigante identificado pelo DBSCAN.
    
    Parâmetros:
    - embeddings_2d: array numpy com os dados 2D
    - clusters_dbscan: array com os rótulos dos clusters do DBSCAN
    - n_clusters_kmeans: número de clusters para o K-means (padrão: 5)
    
    Retorna:
    - dados_cluster_gigante: dados do cluster gigante
    - clusters_kmeans: rótulos dos sub-clusters K-means
    - kmeans_model: modelo K-means treinado
    - cluster_gigante_id: ID do cluster gigante
    """
    
    # 1. Identificar o cluster gigante
    unique_clusters, counts = np.unique(clusters_dbscan, return_counts=True)
    cluster_sizes = dict(zip(unique_clusters, counts))
    
    # Encontrar o cluster com mais pontos (excluindo ruído -1)
    cluster_sizes_no_noise = {k: v for k, v in cluster_sizes.items() if k != -1}
    cluster_gigante_id = max(cluster_sizes_no_noise, key=cluster_sizes_no_noise.get)
    
    print(f"Cluster gigante identificado: Cluster {cluster_gigante_id}")
    print(f"Tamanho: {cluster_sizes_no_noise[cluster_gigante_id]} pontos")
    
    # 2. Filtrar dados do cluster gigante
    mask_cluster_gigante = clusters_dbscan == cluster_gigante_id
    dados_cluster_gigante = embeddings_2d[mask_cluster_gigante]
    
    print(f"Dados filtrados: {dados_cluster_gigante.shape}")
    
    # 3. Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
    clusters_kmeans = kmeans.fit_predict(dados_cluster_gigante)
    
    print(f"K-means aplicado com {n_clusters_kmeans} clusters")
    
    # 4. Analisar resultados
    unique_kmeans_clusters, kmeans_counts = np.unique(clusters_kmeans, return_counts=True)
    print("\nTamanho dos sub-clusters:")
    for cluster_id, size in zip(unique_kmeans_clusters, kmeans_counts):
        print(f"  Sub-cluster {cluster_id}: {size} pontos")
    
    return dados_cluster_gigante, clusters_kmeans, kmeans, cluster_gigante_id

def visualizar_resultados(embeddings_2d, clusters_dbscan, dados_cluster_gigante, 
                         clusters_kmeans, cluster_gigante_id, kmeans_model):
    """
    Cria visualizações dos resultados.
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: DBSCAN original
    ax1 = axes[0]
    scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=clusters_dbscan, cmap='viridis', alpha=0.6, s=1)
    ax1.set_title('DBSCAN - Clusters Identificados')
    ax1.set_xlabel('Dimensão 1')
    ax1.set_ylabel('Dimensão 2')
    plt.colorbar(scatter1, ax=ax1, label='Cluster DBSCAN')
    
    # Plot 2: Cluster gigante destacado
    ax2 = axes[1]
    # Outros clusters em cinza
    mask_outros = clusters_dbscan != cluster_gigante_id
    ax2.scatter(embeddings_2d[mask_outros, 0], embeddings_2d[mask_outros, 1], 
               c='lightgray', alpha=0.3, s=1, label='Outros clusters')
    # Cluster gigante em vermelho
    ax2.scatter(dados_cluster_gigante[:, 0], dados_cluster_gigante[:, 1], 
               c='red', alpha=0.7, s=2, label=f'Cluster gigante ({cluster_gigante_id})')
    ax2.set_title('Cluster Gigante Destacado')
    ax2.set_xlabel('Dimensão 1')
    ax2.set_ylabel('Dimensão 2')
    ax2.legend()
    
    # Plot 3: K-means no cluster gigante
    ax3 = axes[2]
    scatter3 = ax3.scatter(dados_cluster_gigante[:, 0], dados_cluster_gigante[:, 1], 
                          c=clusters_kmeans, cmap='plasma', alpha=0.7, s=2)
    # Centroides
    centroids = kmeans_model.cluster_centers_
    ax3.scatter(centroids[:, 0], centroids[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroides')
    ax3.set_title('K-means no Cluster Gigante')
    ax3.set_xlabel('Dimensão 1')
    ax3.set_ylabel('Dimensão 2')
    plt.colorbar(scatter3, ax=ax3, label='Sub-cluster K-means')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
"""
# Assumindo que você já tem:
# - embeddings_2d: seus dados 2D
# - clusters_dbscan: resultados do DBSCAN

# Aplicar K-means no cluster gigante
dados_cluster_gigante, clusters_kmeans, kmeans_model, cluster_gigante_id = \
    aplicar_kmeans_no_cluster_gigante(embeddings_2d, clusters_dbscan, n_clusters_kmeans=5)

# Visualizar resultados
visualizar_resultados(embeddings_2d, clusters_dbscan, dados_cluster_gigante, 
                     clusters_kmeans, cluster_gigante_id, kmeans_model)

# Criar DataFrame com resultados
df_cluster_gigante = pd.DataFrame(dados_cluster_gigante, columns=['dim1', 'dim2'])
df_cluster_gigante['sub_cluster_kmeans'] = clusters_kmeans
df_cluster_gigante['cluster_dbscan_original'] = cluster_gigante_id

print("DataFrame criado com os sub-clusters K-means!")
""" 