import pandas as pd
import seaborn as sns
import streamlit as st

import matplotlib.pyplot as plt

def app():
    st.title("Distribuição da Qualidade dos Vinhos")
    
    try:
        @st.cache_data
        def load_data():
            try:
                df = pd.read_csv('data/raw/combined_wine_quality.csv', sep=';')
            except FileNotFoundError:
                import os
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_dir = os.path.dirname(script_dir)
                df = pd.read_csv(os.path.join(project_dir, 'data/raw/combined_wine_quality.csv'), sep=';')
            
            if 'quality' not in df.columns:
                columns = [col for col in df.columns if col.lower() == 'quality']
                if columns:
                    df = df.rename(columns={columns[0]: 'quality'})
                elif 'score' in df.columns:
                    df = df.rename(columns={'score': 'quality'})
                elif 'rating' in df.columns:
                    df = df.rename(columns={'rating': 'quality'})
                else:
                    st.warning(f"Não foi possível encontrar a coluna 'quality'. Colunas disponíveis: {df.columns.tolist()}")
                    if len(df.columns) > 0:
                        numeric_cols = df.select_dtypes(include=['int', 'float']).columns
                        if len(numeric_cols) > 0:
                            df['quality'] = df[numeric_cols[0]].copy()
                            st.info(f"Criada coluna 'quality' baseada na coluna '{numeric_cols[0]}'")
            
            return df
        
        df = load_data()
        
        if 'quality' not in df.columns:
            st.error("O dataset não contém a coluna 'quality'.")
            return
        
        st.subheader("Histograma da Qualidade dos Vinhos")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(data=df, x='quality', bins=range(1, 11), kde=True, ax=ax)
        
        ax.set_title("Distribuição da Qualidade dos Vinhos", fontsize=16)
        ax.set_xlabel("Qualidade (escala de 1-10)", fontsize=12)
        ax.set_ylabel("Frequência", fontsize=12)
        ax.set_xticks(range(1, 11))
        
        st.pyplot(fig)
        
        st.subheader("Estatísticas da Qualidade dos Vinhos")
        stats = df['quality'].describe()
        st.write(stats)
        
        st.subheader("Contagem por Nível de Qualidade")
        quality_counts = df['quality'].value_counts().sort_index()
        st.bar_chart(quality_counts)
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar ou processar os dados: {e}")

if __name__ == "__main__":
    app()
