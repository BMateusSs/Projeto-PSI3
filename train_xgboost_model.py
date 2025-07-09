import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import ADASYN
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Carrega e prepara os dados"""
    print("Carregando dados...")
    
    # Carregar dados
    df = pd.read_parquet('data/processed/wine-quality-combined.parquet')
    
    # Preparar variáveis
    df['good_quality'] = (df['quality'] >= 6).astype(int)
    
    # Features
    features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ]
    
    X = df[features]
    y = df['good_quality']
    
    print(f"Shape dos dados: {X.shape}")
    print(f"Distribuição das classes: {y.value_counts().to_dict()}")
    
    return X, y, features

def train_xgboost_with_adasyn():
    """Treina modelo XGBoost com ADASYN"""
    print("\n=== Treinamento do Modelo XGBoost com ADASYN ===")
    
    # Carregar dados
    X, y, features = load_and_prepare_data()
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ADASYN para balanceamento
    print("Aplicando ADASYN...")
    adasyn = ADASYN(random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)
    
    print(f"Após ADASYN - Treino: {X_train_resampled.shape}")
    print(f"Distribuição após ADASYN: {np.bincount(y_train_resampled)}")
    
    # Modelo XGBoost
    print("Treinando XGBoost...")
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
    
    # Avaliação
    print("\n=== Avaliação do Modelo ===")
    
    # Predições
    y_pred_train = xgb_model.predict(X_train_resampled)
    y_pred_test = xgb_model.predict(X_test_scaled)
    
    # Métricas
    print("\n--- Métricas no Conjunto de Treino ---")
    print(classification_report(y_train_resampled, y_pred_train))
    
    print("\n--- Métricas no Conjunto de Teste ---")
    print(classification_report(y_test, y_pred_test))
    
    # AUC-ROC
    y_proba_test = xgb_model.predict_proba(X_test_scaled)[:, 1]
    auc_roc = roc_auc_score(y_test, y_proba_test)
    print(f"\nAUC-ROC no teste: {auc_roc:.4f}")
    
    return xgb_model, scaler, features, X_test_scaled, y_test

def save_model(model, scaler, features):
    """Salva o modelo treinado"""
    print("\n=== Salvando Modelo ===")
    
    # Criar diretório se não existir
    os.makedirs('models', exist_ok=True)
    
    # Dados do modelo
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': features,
        'model_type': 'XGBoost with ADASYN',
        'training_info': {
            'n_estimators': 150,
            'max_depth': 7,
            'learning_rate': 0.2,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    }
    
    # Salvar
    model_path = 'models/xgboost_adasyn_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Modelo salvo em: {model_path}")
    
    # Verificar se foi salvo corretamente
    with open(model_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print("✅ Modelo carregado com sucesso!")
    print(f"Tipo do modelo: {loaded_data['model_type']}")
    print(f"Número de features: {len(loaded_data['feature_names'])}")

def test_prediction():
    """Testa a predição com o modelo salvo"""
    print("\n=== Teste de Predição ===")
    
    # Carregar modelo
    with open('models/xgboost_adasyn_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['feature_names']
    
    # Dados de exemplo
    sample_data = {
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
    
    # Converter para DataFrame
    sample_df = pd.DataFrame([sample_data])
    
    # Normalizar
    sample_scaled = scaler.transform(sample_df)
    
    # Predição
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0]
    
    print(f"Predição: {'Bom' if prediction == 1 else 'Ruim'}")
    print(f"Probabilidade (Ruim): {probability[0]:.3f}")
    print(f"Probabilidade (Bom): {probability[1]:.3f}")
    
    return True

if __name__ == "__main__":
    print("🚀 Iniciando treinamento do modelo XGBoost com ADASYN...")
    
    try:
        # Treinar modelo
        model, scaler, features, X_test, y_test = train_xgboost_with_adasyn()
        
        # Salvar modelo
        save_model(model, scaler, features)
        
        # Testar predição
        test_prediction()
        
        print("\n✅ Treinamento concluído com sucesso!")
        print("O modelo está pronto para uso no Streamlit.")
        
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc() 