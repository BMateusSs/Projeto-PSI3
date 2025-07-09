#!/usr/bin/env python3
"""
Script de teste para o sistema de predição de qualidade de vinhos
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

def test_model_loading():
    """Testa se o modelo pode ser carregado corretamente"""
    print("🔍 Testando carregamento do modelo...")
    
    try:
        model_path = 'models/xgboost_adasyn_model.pkl'
        
        if not os.path.exists(model_path):
            print("❌ Modelo não encontrado. Execute primeiro:")
            print("   python train_xgboost_model.py")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['feature_names']
        
        print(f"✅ Modelo carregado com sucesso!")
        print(f"   Tipo: {model_data.get('model_type', 'N/A')}")
        print(f"   Features: {len(features)}")
        print(f"   Features: {features}")
        
        return True, model, scaler, features
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False, None, None, None

def test_prediction(model, scaler, features):
    """Testa predições com dados de exemplo"""
    print("\n🍷 Testando predições...")
    
    # Dados de exemplo (vinho de qualidade boa)
    sample_data_good = {
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
    
    # Dados de exemplo (vinho de qualidade ruim)
    sample_data_bad = {
        'fixed acidity': 8.1,
        'volatile acidity': 0.56,
        'citric acid': 0.00,
        'residual sugar': 2.2,
        'chlorides': 0.092,
        'free sulfur dioxide': 9.0,
        'total sulfur dioxide': 18.0,
        'density': 0.998,
        'pH': 3.42,
        'sulphates': 0.47,
        'alcohol': 9.4
    }
    
    test_cases = [
        ("Vinho de Qualidade Boa", sample_data_good),
        ("Vinho de Qualidade Ruim", sample_data_bad)
    ]
    
    for name, data in test_cases:
        print(f"\n--- {name} ---")
        
        # Converter para DataFrame
        df = pd.DataFrame([data])
        
        # Verificar se todas as features estão presentes
        missing_features = set(features) - set(df.columns)
        if missing_features:
            print(f"❌ Features faltando: {missing_features}")
            continue
        
        # Normalizar
        df_scaled = scaler.transform(df[features])
        
        # Predição
        prediction = model.predict(df_scaled)[0]
        probability = model.predict_proba(df_scaled)[0]
        
        print(f"Predição: {'🍷 BOM' if prediction == 1 else '❌ RUIM'}")
        print(f"Probabilidade (Ruim): {probability[0]:.3f}")
        print(f"Probabilidade (Bom): {probability[1]:.3f}")
        print(f"Confiança: {max(probability):.1%}")

def test_data_loading():
    """Testa se os dados podem ser carregados"""
    print("\n📊 Testando carregamento de dados...")
    
    try:
        data_path = 'data/processed/wine-quality-combined.parquet'
        
        if not os.path.exists(data_path):
            print("❌ Dados não encontrados.")
            return False
        
        df = pd.read_parquet(data_path)
        print(f"✅ Dados carregados com sucesso!")
        print(f"   Shape: {df.shape}")
        print(f"   Colunas: {list(df.columns)}")
        
        # Verificar se há dados de qualidade
        if 'quality' in df.columns:
            quality_dist = df['quality'].value_counts().sort_index()
            print(f"   Distribuição de qualidade: {quality_dist.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return False

def test_streamlit_pages():
    """Testa se as páginas do Streamlit existem"""
    print("\n🌐 Testando páginas do Streamlit...")
    
    pages_dir = 'pages'
    required_pages = [
        '3_📈_Análise interativa.py',
        '5_🔮_Predição.py',
        '6_📊_Business_Intelligence.py'
    ]
    
    if not os.path.exists(pages_dir):
        print("❌ Diretório 'pages' não encontrado.")
        return False
    
    existing_pages = os.listdir(pages_dir)
    
    for page in required_pages:
        if page in existing_pages:
            print(f"✅ {page}")
        else:
            print(f"❌ {page} - não encontrado")
    
    return True

def main():
    """Função principal de teste"""
    print("🧪 Iniciando testes do sistema de predição...")
    print("=" * 50)
    
    # Teste 1: Carregamento de dados
    data_ok = test_data_loading()
    
    # Teste 2: Carregamento do modelo
    model_ok, model, scaler, features = test_model_loading()
    
    # Teste 3: Predições
    if model_ok:
        test_prediction(model, scaler, features)
    
    # Teste 4: Páginas do Streamlit
    pages_ok = test_streamlit_pages()
    
    # Resumo
    print("\n" + "=" * 50)
    print("📋 RESUMO DOS TESTES:")
    print(f"   Dados: {'✅' if data_ok else '❌'}")
    print(f"   Modelo: {'✅' if model_ok else '❌'}")
    print(f"   Páginas: {'✅' if pages_ok else '❌'}")
    
    if data_ok and model_ok and pages_ok:
        print("\n🎉 Todos os testes passaram! O sistema está pronto para uso.")
        print("\nPara executar o Streamlit:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Alguns testes falharam. Verifique os erros acima.")
        
        if not data_ok:
            print("\n💡 Para resolver problemas com dados:")
            print("   - Verifique se o arquivo 'data/processed/wine-quality-combined.parquet' existe")
        
        if not model_ok:
            print("\n💡 Para resolver problemas com modelo:")
            print("   - Execute: python train_xgboost_model.py")
        
        if not pages_ok:
            print("\n💡 Para resolver problemas com páginas:")
            print("   - Verifique se todos os arquivos .py estão na pasta 'pages'")

if __name__ == "__main__":
    main() 