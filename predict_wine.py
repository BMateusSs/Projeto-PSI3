import pandas as pd
import numpy as np
import joblib

# Carregar o modelo, o scaler e as features importantes
model = joblib.load('svm_wine_model.joblib')
scaler = joblib.load('wine_scaler.joblib')
important_features = joblib.load('wine_features.joblib')

def predict_wine_quality(wine_data):
    """
    Faz previsão da qualidade do vinho.
    
    Args:
        wine_data: DataFrame com as features do vinho
        
    Returns:
        int: 0 para vinho ruim, 1 para vinho bom
        float: probabilidade de ser um vinho bom
    """
    # Selecionar apenas as features importantes
    wine_data = wine_data[important_features]
    
    # Aplicar o scaler
    wine_scaled = scaler.transform(wine_data)
    
    # Fazer a previsão
    prediction = model.predict(wine_scaled)
    probability = model.predict_proba(wine_scaled)[:, 1]
    
    return prediction[0], probability[0]

# Exemplo de uso
if __name__ == "__main__":
    # Carregar dados de exemplo
    df = pd.read_csv('data/raw/combined_wine_quality.csv', sep=';')
    
    # Selecionar uma amostra para teste
    sample = df.drop(['quality', 'wine_type'], axis=1).iloc[0:1]
    
    # Fazer previsão
    prediction, probability = predict_wine_quality(sample)
    
    print(f"\nPrevisão: {'Bom' if prediction == 1 else 'Ruim'}")
    print(f"Probabilidade de ser bom: {probability:.2%}") 