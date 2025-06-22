import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carregar o dataset
# O arquivo Parquet contém as colunas físico-químicas e a coluna 'quality'
df = pd.read_parquet('data/processed/wine-quality-combined.parquet')

# Binarizar a coluna de qualidade: 0 = ruim, 1 = bom
# Considera-se vinho bom se quality >= 6, caso contrário ruim
# (ajuste o limiar conforme necessário)
df['target'] = (df['quality'] >= 6).astype(int)

# Selecionar apenas as colunas numéricas para o modelo
X = df.drop(columns=['type', 'quality', 'target'])
y = df['target']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Instanciar e treinar o modelo Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = gnb.predict(X_test)

# Avaliar o modelo
print('Acurácia:', accuracy_score(y_test, y_pred))
print('\nMatriz de Confusão:')
print(confusion_matrix(y_test, y_pred))
print('\nRelatório de Classificação:')
print(classification_report(y_test, y_pred, target_names=['Ruim', 'Bom'])) 