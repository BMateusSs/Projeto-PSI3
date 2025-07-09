#!/usr/bin/env python3
"""
Script para criar modelo pré-treinado que pode ser incluído no projeto
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import ADASYN
import xgboost as xgb

def create_pretrained_model():
    """Cria modelo pré-treinado para distribuição"""
    print("🚀 Criando modelo pré-treinado...")
    
    # Carregar dados
    df = pd.read_parquet('data/processed/wine-quality-combined.parquet')
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
    
    y_pred_train = xgb_model.predict(X_train_resampled)
    y_pred_test = xgb_model.predict(X_test_scaled)
    
    print("\n--- Métricas no Conjunto de Teste ---")
    print(classification_report(y_test, y_pred_test))
    
    y_proba_test = xgb_model.predict_proba(X_test_scaled)[:, 1]
    auc_roc = roc_auc_score(y_test, y_proba_test)
    print(f"\nAUC-ROC no teste: {auc_roc:.4f}")
    
    # Salvar modelo
    print("\n=== Salvando Modelo ===")
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': xgb_model,
        'scaler': scaler,
        'feature_names': features,
        'model_type': 'XGBoost with ADASYN',
        'training_info': {
            'n_estimators': 150,
            'max_depth': 7,
            'learning_rate': 0.2,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'performance': {
            'auc_roc': auc_roc,
            'test_accuracy': (y_pred_test == y_test).mean()
        }
    }
    
    model_path = 'models/xgboost_adasyn_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Modelo salvo em: {model_path}")
    print(f"Tamanho do arquivo: {os.path.getsize(model_path) / 1024:.1f} KB")
    
    # Teste de carregamento
    print("\n=== Teste de Carregamento ===")
    with open(model_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print("✅ Modelo carregado com sucesso!")
    print(f"Tipo: {loaded_data['model_type']}")
    print(f"Features: {len(loaded_data['feature_names'])}")
    print(f"AUC-ROC: {loaded_data['performance']['auc_roc']:.4f}")
    print(f"Acurácia: {loaded_data['performance']['test_accuracy']:.4f}")
    
    return model_path

if __name__ == "__main__":
    try:
        model_path = create_pretrained_model()
        print(f"\n🎉 Modelo pré-treinado criado com sucesso!")
        print(f"📁 Arquivo: {model_path}")
        print(f"💡 O modelo está pronto para uso no Streamlit!")
        
    except Exception as e:
        print(f"❌ Erro ao criar modelo: {e}")
        import traceback
        traceback.print_exc() 