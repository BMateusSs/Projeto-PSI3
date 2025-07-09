import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Predição de Qualidade de Vinhos")

# Tradução das variáveis
features_en = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
    'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
]
features_pt = [
    'Acidez Fixa', 'Acidez Volátil', 'Ácido Cítrico', 'Açúcar Residual', 'Cloretos',
    'Dióxido de Enxofre Livre', 'Dióxido de Enxofre Total', 'Densidade', 'pH', 'Sulfatos', 'Teor Alcoólico'
]
feature_dict = dict(zip(features_en, features_pt))
feature_dict_inv = {v: k for k, v in feature_dict.items()}

@st.cache_data
def load_data():
    try:
        df = pd.read_parquet('data/processed/wine-quality-combined.parquet')
        df['good_quality'] = (df['quality'] >= 6).astype(int)
        if 'type' in df.columns:
            df = df.rename(columns={'type': 'wine_type'})
        if df['wine_type'].dtype == 'object':
            df['wine_type'] = df['wine_type'].map({'red': 'Tinto', 'white': 'Branco'})
        df['qualidade_binaria'] = df['good_quality'].map({0: 'Ruim', 1: 'Bom'})
        return df
    except FileNotFoundError:
        st.error("❌ Erro: Arquivo de dados não encontrado em 'data/processed/wine-quality-combined.parquet'")
        return None

@st.cache_resource
def load_model():
    try:
        model_path = 'models/xgboost_adasyn_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data['model'], model_data['scaler'], model_data['feature_names']
        else:
            return None, None, None
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None, None, None

def train_model_automatically():
    """Treina o modelo automaticamente se não existir"""
    try:
        st.info("🤖 Treinando modelo automaticamente...")
        
        # Importações necessárias
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, roc_auc_score
        from imblearn.over_sampling import ADASYN
        import xgboost as xgb
        
        # Carregar dados
        df = load_data()
        if df is None:
            return None, None, None
            
        # Preparar dados
        X = df[features_en]
        y = df['good_quality']
        
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ADASYN para balanceamento
        adasyn = ADASYN(random_state=42)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)
        
        # Modelo XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.2,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train_resampled, y_train_resampled)
        
        # Salvar modelo
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': xgb_model,
            'scaler': scaler,
            'feature_names': features_en,
            'model_type': 'XGBoost with ADASYN',
            'training_info': {
                'n_estimators': 150,
                'max_depth': 7,
                'learning_rate': 0.2,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        }
        
        with open('models/xgboost_adasyn_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        st.success("✅ Modelo treinado e salvo com sucesso!")
        return xgb_model, scaler, features_en
        
    except Exception as e:
        st.error(f"❌ Erro ao treinar modelo: {e}")
        return None, None, None

def create_sample_data():
    return {
        'fixed acidity': 7.0,
        'volatile acidity': 0.27,
        'citric acid': 0.36,
        'residual sugar': 20.7,
        'chlorides': 0.045,
        'free sulfur dioxide': 45.0,
        'total sulfur dioxide': 170.0,
        'density': 1.001,
        'pH': 3.00,
        'sulphates': 0.45,
        'alcohol': 8.8
    }

def predict_quality(input_data, model, scaler):
    try:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        return prediction, probability
    except Exception as e:
        st.error(f"❌ Erro na predição: {e}")
        return None, None

def explain_prediction(input_data, model, scaler):
    try:
        # Verificar se SHAP está disponível
        try:
            import shap
        except ImportError:
            st.warning("⚠️ SHAP não está disponível. Instale com: pip install shap")
            return None
            
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)
        
        feature_importance = pd.DataFrame({
            'Feature': [feature_dict.get(f, f) for f in input_df.columns],
            'SHAP_Value': shap_values[0] if len(shap_values) == 2 else shap_values[0]
        })
        feature_importance = feature_importance.sort_values('SHAP_Value', key=abs, ascending=False)
        
        return feature_importance
    except Exception as e:
        st.error(f"❌ Erro na explicação SHAP: {e}")
        return None

st.title("🔮 Predição de Qualidade de Vinhos")
st.markdown("Insira as características do vinho para prever sua qualidade.")

# Carregar dados e modelo
df = load_data()
model, scaler, feature_names = load_model()

# Se o modelo não existe, oferecer para treinar automaticamente
if model is None:
    st.warning("⚠️ Modelo não encontrado.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🤖 Treinar Modelo Automaticamente", type="primary"):
            with st.spinner("Treinando modelo..."):
                model, scaler, feature_names = train_model_automatically()
                if model is not None:
                    st.success("✅ Modelo treinado com sucesso! Recarregue a página para usar.")
                    st.rerun()
    
    with col2:
        st.info("💡 **Alternativas:**\n"
                "1. Clique no botão acima para treinar automaticamente\n"
                "2. Execute manualmente: `python train_xgboost_model.py`\n"
                "3. Verifique se o arquivo existe em: `models/xgboost_adasyn_model.pkl`")
    
    st.stop()

# Sidebar para navegação
page = st.sidebar.selectbox(
    "Escolha a funcionalidade:",
    ["Predição Individual", "Análise em Lote", "Explicações do Modelo", "Exportar Modelo"]
)

if page == "Predição Individual":
    st.header("Predição Individual")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Características do Vinho")
        
        # Campos de entrada
        input_data = {}
        for i, (feat_en, feat_pt) in enumerate(zip(features_en, features_pt)):
            if i % 2 == 0:
                col_a, col_b = st.columns(2)
                with col_a:
                    input_data[feat_en] = st.number_input(
                        feat_pt,
                        value=create_sample_data()[feat_en],
                        format="%.3f",
                        key=f"input_{feat_en}"
                    )
            else:
                with col_b:
                    input_data[feat_en] = st.number_input(
                        feat_pt,
                        value=create_sample_data()[feat_en],
                        format="%.3f",
                        key=f"input_{feat_en}"
                    )
        
        # Botão de predição
        if st.button("🔮 Prever Qualidade", type="primary"):
            prediction, probability = predict_quality(input_data, model, scaler)
            
            if prediction is not None and probability is not None:
                with col2:
                    st.subheader("Resultado da Predição")
                    
                    if prediction == 1:
                        st.success("🍷 **Qualidade: BOM**")
                        st.metric("Probabilidade", f"{probability[1]:.1%}")
                    else:
                        st.error("🍷 **Qualidade: RUIM**")
                        st.metric("Probabilidade", f"{probability[0]:.1%}")
                    
                    st.metric("Confiança", f"{max(probability):.1%}")
                
                # Explicação SHAP
                st.subheader("📊 Explicação da Predição (SHAP)")
                feature_importance = explain_prediction(input_data, model, scaler)
                
                if feature_importance is not None:
                    fig = px.bar(
                        feature_importance,
                        x='SHAP_Value',
                        y='Feature',
                        orientation='h',
                        title="Contribuição das Características para a Predição",
                        color='SHAP_Value',
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explicação textual
                    st.subheader("💡 Explicação Detalhada")
                    top_features = feature_importance.head(3)
                    
                    for _, row in top_features.iterrows():
                        if row['SHAP_Value'] > 0:
                            st.write(f"✅ **{row['Feature']}** contribuiu positivamente para a predição.")
                        else:
                            st.write(f"❌ **{row['Feature']}** contribuiu negativamente para a predição.")

elif page == "Análise em Lote":
    st.header("Análise em Lote")
    
    # Upload de arquivo CSV
    uploaded_file = st.file_uploader("Upload arquivo CSV com características dos vinhos", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Dados carregados:")
            st.dataframe(batch_df.head())
            
            if st.button("🔮 Prever Lote"):
                predictions = []
                probabilities = []
                
                for idx, row in batch_df.iterrows():
                    pred, prob = predict_quality(row.to_dict(), model, scaler)
                    predictions.append(pred)
                    probabilities.append(prob)
                
                batch_df['Predição'] = ['Bom' if p == 1 else 'Ruim' for p in predictions]
                batch_df['Probabilidade_Bom'] = [p[1] for p in probabilities]
                batch_df['Probabilidade_Ruim'] = [p[0] for p in probabilities]
                
                st.subheader("Resultados da Predição")
                st.dataframe(batch_df)
                
                # Download dos resultados
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Resultados",
                    data=csv,
                    file_name="predicoes_vinhos.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {e}")

elif page == "Explicações do Modelo":
    st.header("Explicações do Modelo")
    
    # Feature importance global
    st.subheader("Importância Global das Características")
    
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Característica': [feature_dict.get(f, f) for f in feature_names],
            'Importância': model.feature_importances_
        }).sort_values('Importância', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Importância',
            y='Característica',
            orientation='h',
            title="Importância das Características no Modelo"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # SHAP summary plot
    st.subheader("Análise SHAP Global")
    if df is not None:
        try:
            import shap
            sample_data = df[features_en].sample(min(100, len(df)))
            sample_scaled = scaler.transform(sample_data)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_scaled)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, sample_scaled, feature_names=[feature_dict.get(f, f) for f in features_en], show=False)
            st.pyplot(fig)
        except ImportError:
            st.warning("⚠️ SHAP não está disponível. Instale com: pip install shap")
        except Exception as e:
            st.error(f"❌ Erro na análise SHAP: {e}")

elif page == "Exportar Modelo":
    st.header("Exportar Modelo")
    
    if st.button("💾 Exportar Modelo para Pickle"):
        try:
            os.makedirs('models', exist_ok=True)
            
            model_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'feature_dict': feature_dict
            }
            
            with open('models/xgboost_adasyn_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            st.success("✅ Modelo exportado com sucesso para 'models/xgboost_adasyn_model.pkl'")
            
            # Informações do modelo
            st.subheader("Informações do Modelo")
            st.write(f"Tipo: XGBoost com ADASYN")
            st.write(f"Número de características: {len(feature_names)}")
            st.write(f"Características: {', '.join([feature_dict.get(f, f) for f in feature_names])}")
            
        except Exception as e:
            st.error(f"❌ Erro ao exportar modelo: {e}")

# Footer
st.markdown("---")
st.markdown("**Modelo:** XGBoost com ADASYN | **Dataset:** Vinho Verde (Tinto e Branco)") 