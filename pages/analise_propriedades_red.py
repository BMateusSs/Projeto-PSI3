import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide") 

st.title('Características vs. Qualidade do Vinho Tinto')

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
    st.sidebar.header('Selecione a Característica')
    st.sidebar.write('Escolha uma propriedade para ver como ela se relaciona com a qualidade do vinho.')
    
    chemical_properties = [col for col in df.columns if col != 'Qualidade']
    
    selected_property = st.sidebar.selectbox(
        'Propriedade Química:',
        chemical_properties
    )
    
    if selected_property:
        
        fig_scatter = px.scatter(
            df,
            x=selected_property,
            y='Qualidade',
            color='Qualidade', 
            color_continuous_scale=px.colors.sequential.Viridis, 
            trendline="ols",
            trendline_color_override="red", 
            title=f'{selected_property} vs. Qualidade', 
            labels={selected_property: f'{selected_property}', 
                    'Qualidade': 'Qualidade do Vinho (3 = Baixa, 8 = Alta)'},
            hover_data={selected_property: ':.2f', 'Qualidade': True},
            category_orders={"Qualidade": sorted(df['Qualidade'].unique())}
        )

        fig_scatter.update_layout(
            xaxis_title=f'{selected_property}',
            yaxis_title='Qualidade do Vinho',
            yaxis=dict(tickmode='linear', dtick=1), 
            height=550, 
            width=850, 
            margin=dict(l=40, r=40, t=60, b=40), 
            title_font_size=22,
            coloraxis_showscale=False 
        )
        
        fig_scatter.update_traces(
            hovertemplate="<br>".join([
                "**Dados do Vinho:**",
                f"{selected_property}: %{{x:.2f}}",
                "Qualidade: %{y}",
                "<extra></extra>" 
            ])
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    else:
        st.warning('Nenhuma propriedade química selecionada para visualização.')
else:
    st.warning('Nenhum dado foi carregado. Verifique se o arquivo de dados existe e o caminho está correto.')