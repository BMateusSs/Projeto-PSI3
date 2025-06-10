import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide") 

st.title('Análise de Qualidade de Vinhos Tintos')

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
    tab1, tab2, tab3, tab4 = st.tabs(["Análise Individual", "Matriz de Correlação", "Distribuição de Qualidade", "Análise Comparativa"])
    
    with tab1:
        st.sidebar.header('Selecione a Característica')
        st.sidebar.write('Escolha uma propriedade para ver como ela se relaciona com a qualidade do vinho.')
        
        chemical_properties = [col for col in df.columns if col != 'Qualidade']
        
        selected_property = st.sidebar.selectbox(
            'Propriedade Química:',
            chemical_properties
        )
        
        if selected_property:
            st.subheader(f'Relação entre {selected_property} e Qualidade do Vinho')
            
            fig_scatter = px.scatter(
                df,
                x=selected_property,
                y='Qualidade',
                color='Qualidade', 
                color_continuous_scale=px.colors.sequential.Viridis, 
                trendline="ols",
                trendline_color_override="red", 
                title=f'{selected_property} vs. Qualidade do Vinho', 
                labels={selected_property: f'{selected_property}', 
                        'Qualidade': 'Qualidade do Vinho (3 = Baixa, 8 = Alta)'},
                hover_data={selected_property: ':.2f', 'Qualidade': True},
                category_orders={"Qualidade": sorted(df['Qualidade'].unique())}
            )

            fig_scatter.update_layout(
                xaxis_title=f'{selected_property}',
                yaxis_title='Qualidade do Vinho',
                yaxis=dict(tickmode='linear', dtick=1), 
                height=450, 
                title_font_size=18,
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
            
            fig_box = px.box(
                df, 
                x='Qualidade', 
                y=selected_property,
                color='Qualidade',
                color_discrete_sequence=px.colors.sequential.Viridis,
                title=f'Distribuição de {selected_property} por Qualidade',
                category_orders={"Qualidade": sorted(df['Qualidade'].unique())}
            )
            
            fig_box.update_layout(
                xaxis_title='Qualidade do Vinho',
                yaxis_title=f'{selected_property}',
                height=450,
                title_font_size=18,
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
            
            st.subheader(f'Estatísticas de {selected_property} por Qualidade')
            stats_df = df.groupby('Qualidade')[selected_property].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
            stats_df.columns = ['Qualidade', 'Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']
            stats_df = stats_df.round(3)
            st.dataframe(stats_df, use_container_width=True)

    with tab2:
        st.subheader('Matriz de Correlação entre Características')
        
        corr = df.corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            color_continuous_scale='RdBu_r',
            title='Matriz de Correlação das Características',
            text_auto='.2f',
            aspect="auto"
        )
        
        fig_corr.update_layout(height=700)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader('Top 5 Características Correlacionadas com Qualidade')
        corr_quality = corr['Qualidade'].drop('Qualidade').sort_values(ascending=False)
        top_corr = pd.DataFrame({'Característica': corr_quality.index, 'Correlação': corr_quality.values})
        top_corr = top_corr.iloc[:5].round(3)
        
        fig_top = px.bar(
            top_corr,
            x='Característica',
            y='Correlação',
            color='Correlação',
            color_continuous_scale='Viridis',
            title='Características Mais Correlacionadas com Qualidade'
        )
        
        fig_top.update_layout(height=500)
        st.plotly_chart(fig_top, use_container_width=True)
        
    with tab3:
        st.subheader('Distribuição de Qualidade dos Vinhos')
        
        fig_hist = px.histogram(
            df, 
            x='Qualidade',
            color='Qualidade',
            color_discrete_sequence=px.colors.sequential.Viridis,
            title='Distribuição de Qualidade de Vinhos',
            category_orders={"Qualidade": sorted(df['Qualidade'].unique())},
            text_auto=True
        )
        
        fig_hist.update_layout(
            xaxis_title='Qualidade',
            yaxis_title='Contagem',
            xaxis=dict(tickmode='linear'),
            height=500
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        quality_counts = df['Qualidade'].value_counts().reset_index()
        quality_counts.columns = ['Qualidade', 'Contagem']
        quality_counts['Porcentagem'] = (100 * quality_counts['Contagem'] / quality_counts['Contagem'].sum()).round(1)
        quality_counts['Label'] = quality_counts['Qualidade'].astype(str) + ' (' + quality_counts['Porcentagem'].astype(str) + '%)'
        
        fig_pie = px.pie(
            quality_counts,
            values='Contagem',
            names='Label',
            title='Distribuição Percentual de Qualidade',
            color='Qualidade',
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        
        fig_pie.update_layout(height=500)
        st.plotly_chart(fig_pie, use_container_width=True)
            
        st.subheader('Contagem por Nível de Qualidade')
        st.dataframe(quality_counts[['Qualidade', 'Contagem', 'Porcentagem']], use_container_width=True)
        
    with tab4:
        st.subheader('Análise Comparativa de Características')
        
        selected_properties = st.multiselect(
            'Selecione características para comparar:',
            chemical_properties,
            default=chemical_properties[:3]
        )
        
        if selected_properties:
            qualities = sorted(df['Qualidade'].unique())
            selected_qualities = st.multiselect(
                'Selecione qualidades para comparar:',
                qualities,
                default=[min(qualities), 6, max(qualities)]
            )
            
            if selected_qualities:
                radar_df = pd.DataFrame()
                
                for quality in selected_qualities:
                    temp_df = df[df['Qualidade'] == quality][selected_properties]
                    means = temp_df.mean()
                    radar_df[f'Qualidade {quality}'] = means
                
                radar_df = radar_df.reset_index()
                radar_df.columns = ['Característica'] + [f'Qualidade {q}' for q in selected_qualities]
                
                fig_radar = go.Figure()
                
                for quality in selected_qualities:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_df[f'Qualidade {quality}'],
                        theta=radar_df['Característica'],
                        fill='toself',
                        name=f'Qualidade {quality}'
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                        )
                    ),
                    title='Comparação de Características por Qualidade',
                    height=600
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                st.subheader('Valores Médios por Qualidade')
                st.dataframe(radar_df, use_container_width=True)
else:
    st.warning('Nenhum dado foi carregado. Verifique se o arquivo de dados existe e o caminho está correto.')