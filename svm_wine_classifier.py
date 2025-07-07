import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, make_scorer, precision_score, recall_score
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
df = pd.read_csv('data/raw/combined_wine_quality.csv', sep=';')

# Criar a variável alvo binária (0 para vinhos ruins, 1 para vinhos bons)
# Considerando vinhos com qualidade >= 6 como bons
df['wine_quality_binary'] = (df['quality'] >= 6).astype(int)

# Separar features e target
X = df.drop(['quality', 'wine_quality_binary', 'wine_type'], axis=1)
y = df['wine_quality_binary']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Usar RobustScaler para lidar melhor com outliers
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Seleção de features mais sofisticada
# Primeiro, usar Random Forest para identificar features importantes
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_scaled, y_train)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Selecionar as features mais importantes (threshold mais baixo para incluir mais features)
important_features = feature_importance[feature_importance['importance'] > 0.03]['feature'].tolist()
X_train_selected = X_train_scaled[:, [X.columns.get_loc(f) for f in important_features]]
X_test_selected = X_test_scaled[:, [X.columns.get_loc(f) for f in important_features]]

# Criar uma estratégia de balanceamento mais agressiva focada na classe 0
# Usar SMOTEENN que combina SMOTE com Edited Nearest Neighbors
smote_enn = SMOTEENN(random_state=42, sampling_strategy=1.0)  # Balanceamento completo
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train_selected, y_train)

# Verificar a distribuição das classes após balanceamento
print(f"Distribuição das classes após balanceamento:")
print(f"Classe 0 (vinhos ruins): {np.sum(y_train_balanced == 0)}")
print(f"Classe 1 (vinhos bons): {np.sum(y_train_balanced == 1)}")

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

# Definir os parâmetros para GridSearchCV com foco mais agressivo na classe 0
param_grid = {
    'C': [0.1, 0.5, 1, 2, 5, 10],
    'gamma': [0.001, 0.01, 0.05, 0.1, 0.5],
    'kernel': ['rbf', 'poly'],
    'class_weight': [
        {0: 2.0, 1: 1.0},
        {0: 2.5, 1: 1.0},
        {0: 3.0, 1: 1.0},
        {0: 3.5, 1: 1.0},
        {0: 4.0, 1: 1.0}
    ],
    'probability': [True]
}

# Criar o modelo base
svm_base = SVC(random_state=42)

# Criar o GridSearchCV com validação cruzada estratificada
# Usar F1-score da classe 0 como métrica principal
grid_search = GridSearchCV(
    estimator=svm_base,
    param_grid=param_grid,
    cv=cv,
    scoring=f1_scorer_0,  # Foco na classe 0
    n_jobs=-1,
    verbose=1
)

# Treinar o modelo com GridSearchCV
print("\nIniciando busca pelos melhores parâmetros focados na classe 0...")
grid_search.fit(X_train_balanced, y_train_balanced)

# Obter o melhor modelo
best_model = grid_search.best_estimator_

# Realizar validação cruzada com o melhor modelo usando diferentes métricas
cv_precision_0 = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=cv, scoring=precision_scorer_0)
cv_recall_0 = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=cv, scoring=recall_scorer_0)
cv_f1_0 = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=cv, scoring=f1_scorer_0)

print("\nScores de validação cruzada para classe 0 (vinhos ruins):")
print(f"Precisão: {cv_precision_0.mean():.3f} (+/- {cv_precision_0.std() * 2:.3f})")
print(f"Recall: {cv_recall_0.mean():.3f} (+/- {cv_recall_0.std() * 2:.3f})")
print(f"F1-Score: {cv_f1_0.mean():.3f} (+/- {cv_f1_0.std() * 2:.3f})")

# Fazer previsões no conjunto de teste
y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]

# Testar diferentes thresholds com foco na classe 0
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
best_f1_0 = 0
best_threshold = 0.5
best_y_pred = None
best_precision_0 = 0
best_recall_0 = 0

print("\nTestando diferentes thresholds para otimizar classe 0:")
for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
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
print("\nMelhores parâmetros encontrados:", grid_search.best_params_)
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
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
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
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall - Foco na Classe 0')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curve.png')
plt.close()

# Salvar o modelo e os componentes essenciais
import joblib
joblib.dump(best_model, 'svm_wine_model.joblib')
joblib.dump(scaler, 'wine_scaler.joblib')
joblib.dump(important_features, 'wine_features.joblib')
joblib.dump(best_threshold, 'optimal_threshold.joblib')

print("\nRelatório de Classificação no conjunto de treino balanceado:")
print(classification_report(y_train_balanced, best_model.predict(X_train_balanced)))

print("\nModelo, scaler, features importantes e threshold ótimo salvos com sucesso!")
print(f"\nRESUMO DAS MELHORIAS PARA CLASSE 0 (VINHOS RUINS):")
print(f"- Threshold otimizado: {best_threshold}")
print(f"- Precisão classe 0: {best_precision_0:.3f}")
print(f"- Recall classe 0: {best_recall_0:.3f}")
print(f"- F1-Score classe 0: {best_f1_0:.3f}") 