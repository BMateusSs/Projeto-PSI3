 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM para Classificação de Qualidade de Vinhos
============================================

Este script implementa Support Vector Machine (SVM) para classificar vinhos
como bons ou ruins baseado em características físico-químicas.

Autor: Assistente IA
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def carregar_dados():
    """Carrega e prepara os dados de qualidade de vinhos."""
    print("=== CARREGANDO DADOS ===")
    
    # Carregar dados
    try:
        wine_quality = pd.read_parquet("data/processed/wine-quality.parquet")
        print(f"Dados carregados com sucesso! Shape: {wine_quality.shape}")
    except FileNotFoundError:
        print("Arquivo não encontrado. Verifique se o arquivo wine-quality.parquet existe em data/processed/")
        return None, None, None
    
    # Criar variável alvo binária
    wine_quality['good_quality'] = (wine_quality['quality'] >= 7).astype(int)
    
    # Aplicar One-Hot Encoding na coluna 'type'
    wine_quality = pd.get_dummies(wine_quality, columns=['type'], drop_first=True)
    
    # Separar features e target
    X = wine_quality.drop(['quality', 'good_quality'], axis=1)
    y = wine_quality['good_quality']
    
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"Distribuição das classes: {y.value_counts().to_dict()}")
    
    return X, y, wine_quality

def preparar_dados(X, y):
    """Prepara os dados para treinamento."""
    print("\n=== PREPARANDO DADOS ===")
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Aplicar SMOTE para balancear as classes
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"\nApós SMOTE:")
    print(f"X_train_resampled: {X_train_resampled.shape}")
    print(f"y_train_resampled: {y_train_resampled.shape}")
    print(f"Distribuição das classes após SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    
    return (X_train_scaled, X_test_scaled, X_train_resampled, 
            y_train_resampled, y_train, y_test, scaler)

def treinar_svm_basico(X_train_resampled, y_train_resampled, X_test_scaled, y_test):
    """Treina um modelo SVM básico."""
    print("\n=== TREINANDO SVM BÁSICO ===")
    
    # Modelo SVM básico
    svm_basic = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        probability=True
    )
    
    # Treinar modelo
    svm_basic.fit(X_train_resampled, y_train_resampled)
    
    # Fazer previsões
    y_pred_basic = svm_basic.predict(X_test_scaled)
    y_pred_proba_basic = svm_basic.predict_proba(X_test_scaled)[:, 1]
    
    # Avaliar modelo básico
    print("\n--- Resultados do SVM Básico ---")
    print(classification_report(y_test, y_pred_basic))
    
    # Calcular métricas
    accuracy_basic = accuracy_score(y_test, y_pred_basic)
    precision_basic = precision_score(y_test, y_pred_basic)
    recall_basic = recall_score(y_test, y_pred_basic)
    f1_basic = f1_score(y_test, y_pred_basic)
    auc_basic = roc_auc_score(y_test, y_pred_proba_basic)
    
    print(f"\nMétricas do SVM Básico:")
    print(f"Acurácia: {accuracy_basic:.4f}")
    print(f"Precisão: {precision_basic:.4f}")
    print(f"Recall: {recall_basic:.4f}")
    print(f"F1-Score: {f1_basic:.4f}")
    print(f"AUC-ROC: {auc_basic:.4f}")
    
    return svm_basic, y_pred_basic, y_pred_proba_basic, {
        'accuracy': accuracy_basic,
        'precision': precision_basic,
        'recall': recall_basic,
        'f1': f1_basic,
        'auc': auc_basic
    }

def otimizar_hiperparametros(X_train_resampled, y_train_resampled, X_test_scaled, y_test):
    """Otimiza hiperparâmetros do SVM usando Grid Search."""
    print("\n=== OTIMIZANDO HIPERPARÂMETROS ===")
    
    # Definir grid de parâmetros
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear', 'poly'],
        'class_weight': ['balanced', None]
    }
    
    # Modelo base para Grid Search
    svm_base = SVC(random_state=42, probability=True)
    
    # Grid Search
    grid_search = GridSearchCV(
        estimator=svm_base,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Iniciando Grid Search...")
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"\nMelhores hiperparâmetros: {grid_search.best_params_}")
    print(f"Melhor score CV: {grid_search.best_score_:.4f}")
    
    # Modelo otimizado
    svm_optimized = grid_search.best_estimator_
    
    # Fazer previsões
    y_pred_opt = svm_optimized.predict(X_test_scaled)
    y_pred_proba_opt = svm_optimized.predict_proba(X_test_scaled)[:, 1]
    
    # Avaliar modelo otimizado
    print("\n--- Resultados do SVM Otimizado ---")
    print(classification_report(y_test, y_pred_opt))
    
    # Calcular métricas
    accuracy_opt = accuracy_score(y_test, y_pred_opt)
    precision_opt = precision_score(y_test, y_pred_opt)
    recall_opt = recall_score(y_test, y_pred_opt)
    f1_opt = f1_score(y_test, y_pred_opt)
    auc_opt = roc_auc_score(y_test, y_pred_proba_opt)
    
    print(f"\nMétricas do SVM Otimizado:")
    print(f"Acurácia: {accuracy_opt:.4f}")
    print(f"Precisão: {precision_opt:.4f}")
    print(f"Recall: {recall_opt:.4f}")
    print(f"F1-Score: {f1_opt:.4f}")
    print(f"AUC-ROC: {auc_opt:.4f}")
    
    return svm_optimized, y_pred_opt, y_pred_proba_opt, {
        'accuracy': accuracy_opt,
        'precision': precision_opt,
        'recall': recall_opt,
        'f1': f1_opt,
        'auc': auc_opt
    }

def avaliar_modelo_treino(modelo, X_train_resampled, y_train_resampled, nome_modelo):
    """Avalia o modelo no conjunto de treino."""
    print(f"\n=== AVALIAÇÃO NO TREINO - {nome_modelo} ===")
    
    # Previsões no treino
    y_pred_train = modelo.predict(X_train_resampled)
    y_pred_proba_train = modelo.predict_proba(X_train_resampled)[:, 1]
    
    # Métricas no treino
    accuracy_train = accuracy_score(y_train_resampled, y_pred_train)
    precision_train = precision_score(y_train_resampled, y_pred_train)
    recall_train = recall_score(y_train_resampled, y_pred_train)
    f1_train = f1_score(y_train_resampled, y_pred_train)
    auc_train = roc_auc_score(y_train_resampled, y_pred_proba_train)
    
    print(f"\nMétricas no Treino ({nome_modelo}):")
    print(f"Acurácia: {accuracy_train:.4f}")
    print(f"Precisão: {precision_train:.4f}")
    print(f"Recall: {recall_train:.4f}")
    print(f"F1-Score: {f1_train:.4f}")
    print(f"AUC-ROC: {auc_train:.4f}")
    
    return {
        'accuracy': accuracy_train,
        'precision': precision_train,
        'recall': recall_train,
        'f1': f1_train,
        'auc': auc_train
    }

def criar_tabela_comparativa(metricas_basico, metricas_otimizado, metricas_treino_basico, metricas_treino_otimizado):
    """Cria tabela comparativa dos resultados."""
    print("\n=== TABELA COMPARATIVA DE RESULTADOS ===")
    
    # Criar DataFrame com resultados
    resultados = pd.DataFrame({
        'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC'],
        'SVM Básico (Treino)': [
            metricas_treino_basico['accuracy'],
            metricas_treino_basico['precision'],
            metricas_treino_basico['recall'],
            metricas_treino_basico['f1'],
            metricas_treino_basico['auc']
        ],
        'SVM Básico (Teste)': [
            metricas_basico['accuracy'],
            metricas_basico['precision'],
            metricas_basico['recall'],
            metricas_basico['f1'],
            metricas_basico['auc']
        ],
        'SVM Otimizado (Treino)': [
            metricas_treino_otimizado['accuracy'],
            metricas_treino_otimizado['precision'],
            metricas_treino_otimizado['recall'],
            metricas_treino_otimizado['f1'],
            metricas_treino_otimizado['auc']
        ],
        'SVM Otimizado (Teste)': [
            metricas_otimizado['accuracy'],
            metricas_otimizado['precision'],
            metricas_otimizado['recall'],
            metricas_otimizado['f1'],
            metricas_otimizado['auc']
        ]
    })
    
    # Formatar para exibição
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(resultados.to_string(index=False))
    pd.reset_option('display.float_format')
    
    return resultados

def plotar_resultados(y_test, y_pred_basic, y_pred_proba_basic, y_pred_opt, y_pred_proba_opt):
    """Plota gráficos comparativos dos resultados."""
    print("\n=== GERANDO GRÁFICOS ===")
    
    # Configurar subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Matriz de Confusão - SVM Básico
    cm_basic = confusion_matrix(y_test, y_pred_basic)
    disp_basic = ConfusionMatrixDisplay(confusion_matrix=cm_basic, display_labels=["Ruim (0)", "Bom (1)"])
    disp_basic.plot(ax=axes[0,0], cmap='Blues')
    axes[0,0].set_title('Matriz de Confusão - SVM Básico')
    axes[0,0].grid(False)
    
    # 2. Matriz de Confusão - SVM Otimizado
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    disp_opt = ConfusionMatrixDisplay(confusion_matrix=cm_opt, display_labels=["Ruim (0)", "Bom (1)"])
    disp_opt.plot(ax=axes[0,1], cmap='Blues')
    axes[0,1].set_title('Matriz de Confusão - SVM Otimizado')
    axes[0,1].grid(False)
    
    # 3. Curva ROC - Comparação
    fpr_basic, tpr_basic, _ = roc_curve(y_test, y_pred_proba_basic)
    fpr_opt, tpr_opt, _ = roc_curve(y_test, y_pred_proba_opt)
    auc_basic = roc_auc_score(y_test, y_pred_proba_basic)
    auc_opt = roc_auc_score(y_test, y_pred_proba_opt)
    
    axes[1,0].plot(fpr_basic, tpr_basic, color='blue', lw=2, 
                   label=f'SVM Básico (AUC = {auc_basic:.3f})')
    axes[1,0].plot(fpr_opt, tpr_opt, color='red', lw=2, 
                   label=f'SVM Otimizado (AUC = {auc_opt:.3f})')
    axes[1,0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    axes[1,0].set_xlim([0.0, 1.0])
    axes[1,0].set_ylim([0.0, 1.05])
    axes[1,0].set_xlabel('Taxa de Falsos Positivos')
    axes[1,0].set_ylabel('Taxa de Verdadeiros Positivos')
    axes[1,0].set_title('Curva ROC - Comparação')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Comparação de Métricas
    metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    valores_basico = [
        accuracy_score(y_test, y_pred_basic),
        precision_score(y_test, y_pred_basic),
        recall_score(y_test, y_pred_basic),
        f1_score(y_test, y_pred_basic)
    ]
    valores_opt = [
        accuracy_score(y_test, y_pred_opt),
        precision_score(y_test, y_pred_opt),
        recall_score(y_test, y_pred_opt),
        f1_score(y_test, y_pred_opt)
    ]
    
    x = np.arange(len(metricas))
    width = 0.35
    
    axes[1,1].bar(x - width/2, valores_basico, width, label='SVM Básico', alpha=0.8)
    axes[1,1].bar(x + width/2, valores_opt, width, label='SVM Otimizado', alpha=0.8)
    axes[1,1].set_xlabel('Métricas')
    axes[1,1].set_ylabel('Valor')
    axes[1,1].set_title('Comparação de Métricas')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(metricas)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analise_detalhada_classificacao(y_test, y_pred_basic, y_pred_opt):
    """Análise detalhada da classificação por classe."""
    print("\n=== ANÁLISE DETALHADA POR CLASSE ===")
    
    # Relatórios detalhados
    print("\n--- Relatório Detalhado - SVM Básico ---")
    print(classification_report(y_test, y_pred_basic, target_names=['Vinho Ruim', 'Vinho Bom']))
    
    print("\n--- Relatório Detalhado - SVM Otimizado ---")
    print(classification_report(y_test, y_pred_opt, target_names=['Vinho Ruim', 'Vinho Bom']))
    
    # Análise de erros
    print("\n--- Análise de Erros ---")
    
    # SVM Básico
    cm_basic = confusion_matrix(y_test, y_pred_basic)
    print(f"\nSVM Básico:")
    print(f"Verdadeiros Negativos (Vinho Ruim correto): {cm_basic[0,0]}")
    print(f"Falsos Positivos (Vinho Ruim classificado como Bom): {cm_basic[0,1]}")
    print(f"Falsos Negativos (Vinho Bom classificado como Ruim): {cm_basic[1,0]}")
    print(f"Verdadeiros Positivos (Vinho Bom correto): {cm_basic[1,1]}")
    
    # SVM Otimizado
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    print(f"\nSVM Otimizado:")
    print(f"Verdadeiros Negativos (Vinho Ruim correto): {cm_opt[0,0]}")
    print(f"Falsos Positivos (Vinho Ruim classificado como Bom): {cm_opt[0,1]}")
    print(f"Falsos Negativos (Vinho Bom classificado como Ruim): {cm_opt[1,0]}")
    print(f"Verdadeiros Positivos (Vinho Bom correto): {cm_opt[1,1]}")

def main():
    """Função principal."""
    print("=" * 60)
    print("SVM PARA CLASSIFICAÇÃO DE QUALIDADE DE VINHOS")
    print("=" * 60)
    
    # 1. Carregar dados
    X, y, wine_quality = carregar_dados()
    if X is None:
        return
    
    # 2. Preparar dados
    (X_train_scaled, X_test_scaled, X_train_resampled, 
     y_train_resampled, y_train, y_test, scaler) = preparar_dados(X, y)
    
    # 3. Treinar SVM básico
    svm_basic, y_pred_basic, y_pred_proba_basic, metricas_basico = treinar_svm_basico(
        X_train_resampled, y_train_resampled, X_test_scaled, y_test
    )
    
    # 4. Otimizar hiperparâmetros
    svm_optimized, y_pred_opt, y_pred_proba_opt, metricas_otimizado = otimizar_hiperparametros(
        X_train_resampled, y_train_resampled, X_test_scaled, y_test
    )
    
    # 5. Avaliar no treino
    metricas_treino_basico = avaliar_modelo_treino(svm_basic, X_train_resampled, y_train_resampled, "SVM Básico")
    metricas_treino_otimizado = avaliar_modelo_treino(svm_optimized, X_train_resampled, y_train_resampled, "SVM Otimizado")
    
    # 6. Criar tabela comparativa
    resultados = criar_tabela_comparativa(
        metricas_basico, metricas_otimizado, 
        metricas_treino_basico, metricas_treino_otimizado
    )
    
    # 7. Plotar resultados
    plotar_resultados(y_test, y_pred_basic, y_pred_proba_basic, y_pred_opt, y_pred_proba_opt)
    
    # 8. Análise detalhada
    analise_detalhada_classificacao(y_test, y_pred_basic, y_pred_opt)
    
    print("\n" + "=" * 60)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 60)
    
    # Salvar resultados
    resultados.to_csv('resultados_svm_wine.csv', index=False)
    print("\nResultados salvos em 'resultados_svm_wine.csv'")

if __name__ == "__main__":
    main() 