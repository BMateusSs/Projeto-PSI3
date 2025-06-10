import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Análise de Vinhos Combinados", page_icon="🍷", layout="wide")

st.title("Análise de Vinhos Tintos e Brancos")
st.markdown("Esta página apresenta uma análise das características dos vinhos e sua relação com a qualidade.")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/raw/combined_wine_quality.csv')
        if 'type' not in df.columns:
            if 'is_red' in df.columns:
                df['type'] = df['is_red'].apply(lambda x: 'red' if x == 1 else 'white')
            else:
                try:
                    df_red = pd.read_csv('data/raw/winequality-red.csv')
                    df_red['type'] = 'red'
                    df_white = pd.read_csv('data/raw/winequality-white.csv')
                    df_white['type'] = 'white'
                    df = pd.concat([df_red, df_white], ignore_index=True)
                except Exception as e:
                    st.error(f"Não foi possível carregar os datasets separados: {str(e)}")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")
        return pd.DataFrame()

df = load_data()

if 'type' not in df.columns:
    st.error("O dataset não contém a coluna 'type' e não foi possível criá-la. Verifique se está usando o dataset correto.")
    st.stop()

st.sidebar.header("Filtros")

wine_types = ["Todos"] + sorted(df['type'].unique().tolist())
selected_type = st.sidebar.selectbox("Selecione o tipo de vinho:", wine_types)

if selected_type == "Todos":
    filtered_df = df
else:
    filtered_df = df[df['type'] == selected_type]

st.header("Estatísticas Básicas")

st.subheader("📊 Distribuição da Qualidade por Tipo de Vinho")
st.markdown("Este gráfico mostra como a qualidade dos vinhos está distribuída para cada tipo (tinto ou branco).")
fig = px.histogram(df, x="quality", color="type", barmode="group", 
                 labels={"quality": "Qualidade", "count": "Quantidade", "type": "Tipo de Vinho"},
                 title="Distribuição da Qualidade")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🥧 Proporção de Tipos de Vinho na Base de Dados")
st.markdown("Este gráfico de pizza mostra a proporção entre vinhos tintos e brancos no conjunto de dados.")
wine_counts = df['type'].value_counts().reset_index()
wine_counts.columns = ['Tipo de Vinho', 'Contagem']
fig = px.pie(wine_counts, values='Contagem', names='Tipo de Vinho', title="Proporção de Tipos de Vinho")
st.plotly_chart(fig, use_container_width=True)

st.header("Análise de Características")

features = [col for col in filtered_df.columns if col not in ['quality', 'type']]
selected_feature = st.selectbox("Selecione uma característica para analisar:", features)

st.subheader(f"📈 Relação entre {selected_feature} e Qualidade")
st.markdown(f"Este boxplot mostra como o valor de {selected_feature} varia de acordo com a qualidade do vinho.")
fig = px.box(filtered_df, x="quality", y=selected_feature, color="type",
            labels={"quality": "Qualidade", selected_feature: selected_feature.capitalize(), "type": "Tipo de Vinho"},
            title=f"Boxplot: {selected_feature.capitalize()} vs Qualidade")
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"📊 Distribuição de {selected_feature} por Tipo de Vinho")
st.markdown(f"Este histograma mostra como {selected_feature} está distribuído entre os tipos de vinho.")
fig = px.histogram(filtered_df, x=selected_feature, color="type", marginal="box",
                 labels={selected_feature: selected_feature.capitalize(), "count": "Contagem", "type": "Tipo de Vinho"},
                 title=f"Distribuição de {selected_feature.capitalize()}")
st.plotly_chart(fig, use_container_width=True)

st.header("Correlação entre Variáveis")

st.subheader("🔥 Mapa de Calor de Correlação")
st.markdown("Este mapa de calor mostra o grau de correlação entre as diferentes características dos vinhos.")

if selected_type == "Todos":
    st.warning("Para o mapa de calor, selecione um tipo específico de vinho no filtro lateral.")
    corr_df = df[df['type'] == df['type'].unique()[0]].drop(columns=['type'])
else:
    corr_df = filtered_df.drop(columns=['type'])

numeric_cols = corr_df.select_dtypes(include=[np.number]).columns.tolist()
if 'quality' in numeric_cols:
    numeric_cols.remove('quality') 

corr = corr_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, 
            fmt='.2f',  
            annot_kws={"size": 9},  
            mask=mask,  
            square=True) 

plt.title(f"Mapa de Calor de Correlação - {selected_type}")
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9) 
plt.tight_layout() 
st.pyplot(fig)

st.subheader("🔍 Relação entre Duas Características")
st.markdown("Este gráfico de dispersão permite analisar a relação entre duas características selecionadas.")
feature1 = st.selectbox("Selecione a primeira característica:", features, index=0)
feature2 = st.selectbox("Selecione a segunda característica:", features, index=1 if len(features) > 1 else 0)

fig = px.scatter(filtered_df, x=feature1, y=feature2, color="quality", hover_data=["type"],
               labels={feature1: feature1.capitalize(), feature2: feature2.capitalize(), "quality": "Qualidade", "type": "Tipo de Vinho"},
               title=f"Relação entre {feature1.capitalize()} e {feature2.capitalize()}")
st.plotly_chart(fig, use_container_width=True)