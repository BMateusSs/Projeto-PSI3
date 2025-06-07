import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Análise de Vinhos Brancos", layout="wide")

st.title('Análise de Qualidade dos Vinhos Brancos')
st.markdown("""
Esta página apresenta uma análise exploratória do dataset de qualidade de vinhos brancos.
Explore as características físico-químicas e sua relação com a qualidade dos vinhos.
""")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/raw/winequality-white.csv', sep=';')
        if 'quality' not in df.columns:
            st.error("A coluna 'quality' não foi encontrada no dataset.")
        return df
    except FileNotFoundError:
        st.error("Arquivo do dataset não encontrado. Certifique-se de que 'winequality-white.csv' está no diretório de dados.")
        return None

data = load_data()

if data is not None:
    st.subheader('Visão Geral do Dataset')
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Número de amostras:** {data.shape[0]}")
        st.write(f"**Número de características:** {data.shape[1]-1}")
    with col2:
        st.write(f"**Qualidade média dos vinhos:** {data['quality'].mean():.2f}")
        st.write(f"**Faixa de qualidade:** {data['quality'].min()} a {data['quality'].max()}")
    
    st.subheader('Resumo Estatístico')
    st.dataframe(data.describe())
    
    tab1, tab2, tab3 = st.tabs(["Distribuição de Qualidade", "Histogramas", "Análise de Correlação"])
    
    with tab1:
        st.subheader('Distribuição de Qualidade dos Vinhos Brancos')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        quality_counts = data['quality'].value_counts().sort_index()
        bars = sns.barplot(x=quality_counts.index, y=quality_counts.values, ax=ax, palette='viridis')
        
        for i, bar in enumerate(bars.patches):
            bars.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + 20, 
                    f'{quality_counts.values[i]}',
                    ha='center', va='bottom')
        
        ax.set_title('Distribuição de Qualidade dos Vinhos Brancos')
        ax.set_xlabel('Pontuação de Qualidade')
        ax.set_ylabel('Número de Vinhos')
        st.pyplot(fig)
        
        st.subheader('Proporção por Categoria de Qualidade')
        
        def categorize_quality(quality):
            if quality <= 4:
                return 'Baixa (3-4)'
            elif quality <= 6:
                return 'Média (5-6)'
            else:
                return 'Alta (7-9)'
        
        data['quality_category'] = data['quality'].apply(categorize_quality)
        
        fig = px.pie(data, names='quality_category', title='Proporção por Categoria de Qualidade',
                    color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader('Distribuição das Características Físico-Químicas')
        
        all_features = [col for col in data.columns if col != 'quality_category']
        selected_features = st.multiselect('Selecione características para histogramas:', 
                                        all_features, 
                                        default=['alcohol', 'fixed acidity', 'residual sugar'])
        
        if selected_features:
            n_bins = 30
            
            for feature in selected_features:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data[feature], bins=n_bins, kde=True, ax=ax)
                ax.set_title(f'Distribuição de {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequência')
                st.pyplot(fig)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Média", f"{data[feature].mean():.2f}")
                col2.metric("Mediana", f"{data[feature].median():.2f}")
                col3.metric("Desvio Padrão", f"{data[feature].std():.2f}")
                col4.metric("IQR", f"{data[feature].quantile(0.75) - data[feature].quantile(0.25):.2f}")
                
                st.markdown("---")
    
    with tab3:
        st.subheader('Análise de Correlação com a Qualidade')
        

        numeric_data = data.select_dtypes(include=[np.number])
        correlations = numeric_data.corr()['quality'].drop('quality').sort_values(ascending=False)
        
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title='Correlação das Características com a Qualidade do Vinho',
            labels={'x': 'Coeficiente de Correlação', 'y': 'Características'},
            color=correlations.values,
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1]
        )
        st.subheader('Matriz de Correlação Completa')
        fig = px.imshow(
            numeric_data.corr(),
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Matriz de Correlação entre Características',
            labels=dict(x="Características", y="Características", color="Correlação")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader('Características Mais Importantes para a Qualidade')
        st.write("""
        Para vinhos brancos, as características mais correlacionadas com a qualidade são:
        - **Álcool**: Correlação positiva - vinhos com maior teor alcoólico tendem a ter melhor qualidade
        - **Densidade**: Correlação negativa - vinhos menos densos geralmente têm melhor qualidade
        - **Acidez volátil**: Correlação negativa - vinhos com menor acidez volátil tendem a ser melhores
        - **Dióxido de enxofre livre**: Correlação positiva - níveis adequados contribuem para preservação
        """)
