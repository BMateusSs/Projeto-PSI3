import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="Painel de Predição de Qualidade do Vinho", layout="centered")
st.title("Painel Interativo de Predição de Qualidade de Vinho")

st.markdown("Este painel permite carregar um modelo `.pkl`, inserir características de um vinho e prever sua qualidade.")

st.sidebar.header("1. Carregue o modelo (.pkl)")
modelo_pkl = st.sidebar.file_uploader("Escolha um arquivo de modelo (.pkl)", type=["pkl"])

modelo = None
if modelo_pkl is not None:
    modelo = joblib.load(modelo_pkl)
    st.sidebar.success("Modelo carregado com sucesso!")
else:
    st.sidebar.warning("Carregue um modelo para continuar.")

st.sidebar.header("2. Insira os dados do vinho")
campos = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
          'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
          'pH', 'sulphates', 'alcohol', 'type_white']

valores_padrao = {
    'fixed acidity': 7.0,
    'volatile acidity': 0.3,
    'citric acid': 0.3,
    'residual sugar': 6.0,
    'chlorides': 0.045,
    'free sulfur dioxide': 30.0,
    'total sulfur dioxide': 115.0,
    'density': 0.994,
    'pH': 3.2,
    'sulphates': 0.5,
    'alcohol': 10.5,
    'type_white': True
}

entrada = {}

for campo in campos:
    if campo == 'type_white':
        entrada[campo] = st.sidebar.selectbox("Tipo de vinho", ["Tinto", "Branco"],
                                              index=1 if valores_padrao[campo] else 0) == "Branco"
    else:
        entrada[campo] = st.sidebar.number_input(campo, value=valores_padrao[campo], step=0.1)

if st.sidebar.button("Classificar"):
    if modelo is None:
        st.error("Nenhum modelo carregado. Carregue um modelo .pkl na barra lateral.")
    else:
        dados_usuario = pd.DataFrame([entrada])
        dados_usuario['type_white'] = dados_usuario['type_white'].astype(int)

        try:
            predicao = modelo.predict(dados_usuario)[0]
            probabilidade = modelo.predict_proba(dados_usuario)[0][1] if hasattr(modelo, 'predict_proba') else None

            classe = "Vinho de **Boa Qualidade**" if predicao == 1 else "Vinho de **Má Qualidade**"
            st.subheader("Resultado da Predição")
            st.markdown(f"**Previsão:** {classe}")

            if probabilidade is not None:
                st.markdown(f"**Probabilidade de ser bom:** {probabilidade*100:.2f}%")
        except Exception as e:
            st.error(f"Erro ao realizar a predição: {e}")
else:
    st.info("Insira os dados e clique em 'Classificar' para ver o resultado.")
