import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

st.title('Distribuição das pontuações de qualidade dos vinhos Tinto')

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/raw/winequality-red.csv')
        
        traducoes = {
            'fixed acidity': 'Acidez Fixa',
            'volatile acidity': 'Acidez Volátil',
            'citric acid': 'Ácido Cítrico',
            'residual sugar': 'Açúcar Residual',
            'chlorides': 'Cloretos',
            'free sulfur dioxide': 'Dióxido de Enxofre Livre',
            'total sulfur dioxide': 'Dióxido de Enxofre Total',
            'density': 'Densidade',
            'pH': 'pH',
            'sulphates': 'Sulfatos',
            'alcohol': 'Álcool',
            'quality': 'Qualidade'
        }
    
        df = df.rename(columns=traducoes)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

df = load_data()

if df is not None:
    st.sidebar.header('Sobre o Histograma')
    st.sidebar.write('Este histograma mostra a distribuição das pontuações de qualidade dos vinhos Tintos.')
    st.sidebar.write('As pontuações são agrupadas em intervalos para facilitar a visualização.')
    
    st.subheader('Histograma de Pontuações de Qualidade')
    
    score_column = 'Qualidade'
    
    plt.style.use('ggplot')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bin_edges = np.arange(df[score_column].min() - 0.5, df[score_column].max() + 1.5, 1)
    
    bars = sns.histplot(df[score_column], bins=bin_edges, kde=False, ax=ax, color='#3498db', edgecolor='black')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 5,
                    int(height), ha="center", fontsize=10, fontweight='bold')
    
    ax.set_title('Distribuição de Pontuações de Qualidade dos Vinhos', fontsize=16, fontweight='bold')
    ax.set_xlabel('Pontuação de Qualidade', fontsize=12)
    ax.set_ylabel('Número de Vinhos', fontsize=12)
    
    plt.xticks(np.arange(df[score_column].min(), df[score_column].max()+1, 1))
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.subheader('Estatísticas Resumidas')
    
    stats = df[score_column].describe().round(2)
    stats_df = pd.DataFrame({
        'Estatística': ['Contagem', 'Média', 'Desvio Padrão', 'Mínimo', 'Primeiro Quartil', 'Mediana', 'Terceiro Quartil', 'Máximo'],
        'Valor': stats.values
    })
    st.table(stats_df)
    
    st.subheader('Contagem por Pontuação')
    score_counts = df[score_column].value_counts().sort_index()
    score_counts_df = pd.DataFrame({
        score_column: score_counts.index,
        'Quantidade': score_counts.values
    })
    
    import plotly.express as px
    
    fig2 = px.bar(
        score_counts_df, 
        x=score_column, 
        y='Quantidade',
        text='Quantidade',
        color='Quantidade',
        color_continuous_scale='Viridis',
        title='Quantidade de Vinhos por Pontuação de Qualidade'
    )
    
    fig2.update_layout(
        xaxis_title='Pontuação de Qualidade',
        yaxis_title='Quantidade de Vinhos',
        xaxis=dict(tickmode='linear'),
        hovermode='closest',
        height=500,
        width=800,
    )
    
    fig2.update_traces(
        textposition='outside',
        textfont=dict(size=14, color='black'),
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader('Distribuição Percentual por Pontuação')
    
    fig3 = px.pie(
        score_counts_df, 
        values='Quantidade', 
        names=score_column,
        title='Distribuição Percentual das Pontuações',
        color=score_column,
        color_discrete_sequence=px.colors.sequential.Plasma_r
    )
    
    fig3.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hoverinfo='label+percent+value'
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
else:
    st.warning('Nenhum dado foi carregado. Verifique se o arquivo de dados existe.')