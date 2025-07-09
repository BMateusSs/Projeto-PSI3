#!/usr/bin/env python3
"""
Script para testar o modelo pré-treinado
"""

import pickle
import pandas as pd

def test_model():
    """Testa o modelo pré-treinado"""
    try:
        # Carregar modelo
        with open('models/xgboost_adasyn_model.pkl', 'rb') as f:
            data = pickle.load(f)
        
        print("✅ Modelo carregado com sucesso!")
        print(f"Tipo: {data['model_type']}")
        print(f"Features: {len(data['feature_names'])}")
        print(f"AUC-ROC: {data['performance']['auc_roc']:.4f}")
        print(f"Acurácia: {data['performance']['test_accuracy']:.4f}")
        
        # Teste de predição
        model = data['model']
        scaler = data['scaler']
        
        # Dados de teste
        test_data = {
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
        
        # Fazer predição
        input_df = pd.DataFrame([test_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        print(f"\n🧪 Teste de Predição:")
        print(f"Predição: {'Bom' if prediction == 1 else 'Ruim'}")
        print(f"Probabilidade (Ruim): {probability[0]:.3f}")
        print(f"Probabilidade (Bom): {probability[1]:.3f}")
        
        print("\n🎉 Modelo funcionando perfeitamente!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar modelo: {e}")
        return False

if __name__ == "__main__":
    test_model() 