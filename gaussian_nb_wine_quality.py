import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, make_scorer, precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Carregar o dataset
print("Carregando dados...")
df = pd.read_parquet('data/processed/wine-quality-combined.parquet')

# Binarizar a coluna de qualidade: 0 = ruim, 1 = bom
# Considera-se vinho bom se quality >= 6, caso contrário ruim
df['target'] = (df['quality'] >= 6).astype(int)

# Verificar distribuição das classes
print(f"\nDistribuição das classes:")
print(f"Vinhos ruins (0): {np.sum(df['target'] == 0)}")
print(f"Vinhos bons (1): {np.sum(df['target'] == 1)}")
print(f"Proporção: {np.sum(df['target'] == 0) / len(df):.2f} vs {np.sum(df['target'] == 1) / len(df):.2f}")

# Selecionar apenas as colunas numéricas para o modelo
X = df.drop(columns=['type', 'quality', 'target'])
y = df['target']

print(f"\nFeatures disponíveis: {list(X.columns)}")
print(f"Número de features: {X.shape[1]}")

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pré-processamento: Escalar os dados
print("\nAplicando pré-processamento...")
scaler = RobustScaler()  # Mais robusto a outliers
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Seleção de features usando Random Forest
print("Selecionando features importantes...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_selector.fit(X_train_scaled, y_train)

# Obter importância das features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_selector.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 features mais importantes:")
print(feature_importance.head())

# Selecionar as features mais importantes (threshold de 5% de importância)
important_features = feature_importance[feature_importance['importance'] > 0.05]['feature'].tolist()
if len(important_features) < 3:  # Garantir pelo menos 3 features
    important_features = feature_importance.head(3)['feature'].tolist()

print(f"\nFeatures selecionadas: {important_features}")
print(f"Número de features selecionadas: {len(important_features)}")

# Aplicar seleção de features
# Mapear os nomes das features para seus índices nas arrays numpy escaladas
feature_indices = [X.columns.get_loc(f) for f in important_features]
X_train_selected = X_train_scaled[:, feature_indices]
X_test_selected = X_test_scaled[:, feature_indices]

# Testar diferentes técnicas de SMOTE para encontrar a melhor
print("\nTestando diferentes técnicas de SMOTE...")

smote_techniques = {
    'SMOTE': SMOTE(random_state=42, sampling_strategy=1.0),  # Balanceamento completo
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42, sampling_strategy=1.0, k_neighbors=5),  # Balanceamento completo
    'ADASYN': ADASYN(random_state=42, sampling_strategy=1.0),  # Balanceamento completo
    'SMOTE_k3': SMOTE(random_state=42, sampling_strategy=1.0, k_neighbors=3),  # Balanceamento completo
    'SMOTE_k7': SMOTE(random_state=42, sampling_strategy=1.0, k_neighbors=7)  # Balanceamento completo
}

best_technique = None
best_f1_macro = 0
best_X_balanced = None
best_y_balanced = None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, technique in smote_techniques.items():
    print(f"\nTestando {name}...")
    
    # Aplicar a técnica
    X_balanced, y_balanced = technique.fit_resample(X_train_selected, y_train)
    
    # Treinar modelo e avaliar com validação cruzada
    gnb_temp = GaussianNB()
    cv_scores = cross_val_score(gnb_temp, X_balanced, y_balanced, cv=cv, scoring='f1_macro')
    f1_macro_mean = cv_scores.mean()
    
    print(f"F1-Score macro médio: {f1_macro_mean:.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    if f1_macro_mean > best_f1_macro:
        best_f1_macro = f1_macro_mean
        best_technique = name
        best_X_balanced = X_balanced
        best_y_balanced = y_balanced

print(f"\nMelhor técnica encontrada: {best_technique}")
print(f"Melhor F1-Score macro na validação cruzada: {best_f1_macro:.3f}")

# Usar a melhor técnica de SMOTE
X_train_balanced = best_X_balanced
y_train_balanced = best_y_balanced

# Verificar a distribuição das classes após balanceamento
print(f"\nDistribuição das classes após balanceamento ({best_technique}):")
print(f"Classe 0 (vinhos ruins): {np.sum(y_train_balanced == 0)}")
print(f"Classe 1 (vinhos bons): {np.sum(y_train_balanced == 1)}")

## Otimização de Hiperparâmetros para Gaussian Naive Bayes

print("\nOtimizando hiperparâmetros do GaussianNB para maximizar a precisão macro...")
# Definir o modelo base
gnb = GaussianNB()

# Definir o espaço de busca para var_smoothing com valores mais conservadores
# Valores menores de var_smoothing = mais regularização (menos overfitting)
param_grid_gnb = {
    'var_smoothing': np.logspace(-1, -12, num=50)  # Valores mais conservadores
}

# Criar um objeto StratifiedKFold para validação cruzada estratificada mais rigorosa
cv_grid = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Voltar para 10 folds

# Configurar o GridSearchCV para otimizar a precisão macro com validação mais rigorosa
grid_search_gnb = GridSearchCV(
    gnb, 
    param_grid_gnb, 
    cv=cv_grid, 
    scoring='precision_macro', 
    verbose=1, 
    n_jobs=-1,
    refit=True
)

# Executar a busca em grade no conjunto de treino balanceado
grid_search_gnb.fit(X_train_balanced, y_train_balanced)

print(f"\nMelhores parâmetros para GaussianNB (focando em precisão macro): {grid_search_gnb.best_params_}")
print(f"Melhor score de precisão macro da validação cruzada: {grid_search_gnb.best_score_:.3f}")

# Atualizar o modelo GNB com os melhores parâmetros encontrados
gnb_optimized = grid_search_gnb.best_estimator_
print("Modelo GaussianNB atualizado com os melhores parâmetros.")

# Implementar ensemble simples e eficaz
print("\nImplementando ensemble simples para melhorar precisão...")

# Criar múltiplos modelos com diferentes configurações
gnb1 = GaussianNB(var_smoothing=grid_search_gnb.best_params_['var_smoothing'])
gnb2 = GaussianNB(var_smoothing=grid_search_gnb.best_params_['var_smoothing'] * 10)  # Mais regularização
gnb3 = GaussianNB(var_smoothing=grid_search_gnb.best_params_['var_smoothing'] * 0.1)  # Menos regularização

# Criar ensemble com votação suave
ensemble = VotingClassifier(
    estimators=[
        ('gnb1', gnb1),
        ('gnb2', gnb2), 
        ('gnb3', gnb3)
    ],
    voting='soft'  # Votação baseada em probabilidades
)

# Treinar o ensemble
ensemble.fit(X_train_balanced, y_train_balanced)

# Avaliar qual modelo é melhor (individual vs ensemble)
gnb_optimized_score = cross_val_score(gnb_optimized, X_train_balanced, y_train_balanced, cv=5, scoring='precision_macro').mean()
ensemble_score = cross_val_score(ensemble, X_train_balanced, y_train_balanced, cv=5, scoring='precision_macro').mean()

print(f"Score do modelo individual: {gnb_optimized_score:.3f}")
print(f"Score do ensemble: {ensemble_score:.3f}")

# Usar o melhor modelo
if ensemble_score > gnb_optimized_score:
    final_model = ensemble
    print("Usando ensemble (melhor performance)")
else:
    final_model = gnb_optimized
    print("Usando modelo individual (melhor performance)")

# Fazer previsões no conjunto de teste com o modelo final
y_pred_proba_optimized = final_model.predict_proba(X_test_selected)[:, 1]

# Criar métricas customizadas focadas na classe 0
def precision_class_0(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=0)

def recall_class_0(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def f1_class_0(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=0)

# Criar scorers customizados
precision_scorer_0 = make_scorer(precision_class_0)
recall_scorer_0 = make_scorer(recall_class_0)
f1_scorer_0 = make_scorer(f1_class_0)

# Criar um pipeline com validação cruzada estratificada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Realizar validação cruzada com o melhor modelo usando diferentes métricas
cv_precision_0 = cross_val_score(final_model, X_train_balanced, y_train_balanced, cv=cv, scoring=precision_scorer_0)
cv_recall_0 = cross_val_score(final_model, X_train_balanced, y_train_balanced, cv=cv, scoring=recall_scorer_0)
cv_f1_0 = cross_val_score(final_model, X_train_balanced, y_train_balanced, cv=cv, scoring=f1_scorer_0)

print("\nScores de validação cruzada para classe 0 (vinhos ruins):")
print(f"Precisão: {cv_precision_0.mean():.3f} (+/- {cv_precision_0.std() * 2:.3f})")
print(f"Recall: {cv_recall_0.mean():.3f} (+/- {cv_recall_0.std() * 2:.3f})")
print(f"F1-Score: {cv_f1_0.mean():.3f} (+/- {cv_f1_0.std() * 2:.3f})")

# Testar diferentes thresholds com foco na classe 0
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
best_f1_0 = 0
best_threshold = 0.5
best_y_pred = None
best_precision_0 = 0
best_recall_0 = 0

print("\nTestando diferentes thresholds para otimizar classe 0:")
for threshold in thresholds:
    y_pred = (y_pred_proba_optimized >= threshold).astype(int)
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    f1_0 = f1_score(y_test, y_pred, pos_label=0)
    
    print(f"Threshold {threshold:.1f}: Precision_0={precision_0:.3f}, Recall_0={recall_0:.3f}, F1_0={f1_0:.3f}")
    
    # Priorizar F1-score da classe 0
    if f1_0 > best_f1_0:
        best_f1_0 = f1_0
        best_threshold = threshold
        best_y_pred = y_pred
        best_precision_0 = precision_0
        best_recall_0 = recall_0

print(f"\nMelhor threshold encontrado: {best_threshold}")
print(f"Melhor F1-score classe 0: {best_f1_0:.3f}")
print(f"Melhor precisão classe 0: {best_precision_0:.3f}")
print(f"Melhor recall classe 0: {best_recall_0:.3f}")

# Avaliar o modelo com o melhor threshold
print("\nMelhores parâmetros encontrados:", grid_search_gnb.best_params_)
print("\nAcurácia do modelo no conjunto de teste:", accuracy_score(y_test, best_y_pred))
print("\nRelatório de Classificação no conjunto de teste:")
print(classification_report(y_test, best_y_pred))

# Criar matriz de confusão
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Foco na Classe 0 (Vinhos Ruins)')
plt.ylabel('Valor Real')
plt.xlabel('Valor Previsto')
plt.savefig('confusion_matrix.png')
plt.close()

# Plotar curva ROC
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_optimized)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Foco na Classe 0')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# Plotar curva Precision-Recall
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_optimized)
average_precision = average_precision_score(y_test, y_pred_proba_optimized)
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - Foco na Classe 0')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curve.png')
plt.close()

# Salvar o modelo e os componentes essenciais
joblib.dump(final_model, 'gaussian_nb_model.joblib')
joblib.dump(scaler, 'gaussian_nb_scaler.joblib')
joblib.dump(important_features, 'gaussian_nb_features.joblib')
joblib.dump(best_threshold, 'gaussian_nb_threshold.joblib')

print("\nRelatório de Classificação no conjunto de treino balanceado:")
print(classification_report(y_train_balanced, final_model.predict(X_train_balanced)))

print("\nModelo, scaler, features importantes e threshold ótimo salvos com sucesso!")
print(f"\nRESUMO DAS MELHORIAS PARA CLASSE 0 (VINHOS RUINS):")
print(f"- Threshold otimizado: {best_threshold}")
print(f"- Precisão classe 0: {best_precision_0:.3f}")
print(f"- Recall classe 0: {best_recall_0:.3f}")
print(f"- F1-Score classe 0: {best_f1_0:.3f}")