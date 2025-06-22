import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

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
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Selecionar as features mais importantes
important_features = feature_importance[feature_importance['importance'] > 0.05]['feature'].tolist()
X_train_selected = X_train_scaled[:, [X.columns.get_loc(f) for f in important_features]]
X_test_selected = X_test_scaled[:, [X.columns.get_loc(f) for f in important_features]]

# Criar uma estratégia de balanceamento mais focada na classe 0
borderline_smote = BorderlineSMOTE(random_state=42, sampling_strategy=0.8)
X_train_balanced, y_train_balanced = borderline_smote.fit_resample(X_train_selected, y_train)

# Limpar ruído com TomekLinks
tomek = TomekLinks()
X_train_balanced, y_train_balanced = tomek.fit_resample(X_train_balanced, y_train_balanced)

# Criar um pipeline com validação cruzada estratificada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Definir os parâmetros para GridSearchCV com foco na classe 0
param_grid = {
    'C': [0.5, 1, 2, 3],
    'gamma': [0.01, 0.05, 0.1],
    'kernel': ['rbf'],
    'class_weight': [
        {0: 1.5, 1: 1.0},
        {0: 1.8, 1: 1.0},
        {0: 2.0, 1: 1.0}
    ],
    'probability': [True]
}

# Criar o modelo base
svm_base = SVC(random_state=42)

# Criar o GridSearchCV com validação cruzada estratificada
grid_search = GridSearchCV(
    estimator=svm_base,
    param_grid=param_grid,
    cv=cv,
    scoring='precision',
    n_jobs=-1,
    verbose=1
)

# Treinar o modelo com GridSearchCV
print("\nIniciando busca pelos melhores parâmetros...")
grid_search.fit(X_train_balanced, y_train_balanced)

# Obter o melhor modelo
best_model = grid_search.best_estimator_

# Realizar validação cruzada com o melhor modelo
cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=cv, scoring='precision')
print("\nScores de validação cruzada (Precisão):", cv_scores)
print("Média dos scores de validação cruzada:", cv_scores.mean())
print("Desvio padrão dos scores de validação cruzada:", cv_scores.std())

# Fazer previsões no conjunto de teste
y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]

# Testar diferentes thresholds
thresholds = [0.4, 0.5, 0.6, 0.7]
best_f1 = 0
best_threshold = 0.5
best_y_pred = None

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_y_pred = y_pred

print(f"\nMelhor threshold encontrado: {best_threshold}")

# Avaliar o modelo com o melhor threshold
print("\nMelhores parâmetros encontrados:", grid_search.best_params_)
print("\nAcurácia do modelo no conjunto de teste:", accuracy_score(y_test, best_y_pred))
print("\nRelatório de Classificação no conjunto de teste:")
print(classification_report(y_test, best_y_pred))

# Criar matriz de confusão
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
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
plt.title('Curva ROC')
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
plt.title('Curva Precision-Recall')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curve.png')
plt.close()

# Salvar o modelo e os componentes essenciais
import joblib
joblib.dump(best_model, 'svm_wine_model.joblib')
joblib.dump(scaler, 'wine_scaler.joblib')
joblib.dump(important_features, 'wine_features.joblib')

print(classification_report(y_train_balanced, best_model.predict(X_train_balanced)))

print("\nModelo, scaler e features importantes salvos com sucesso!") 