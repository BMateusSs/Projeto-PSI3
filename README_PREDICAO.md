# 🍷 Sistema de Predição de Qualidade de Vinhos

## 📋 Visão Geral

Este sistema permite prever a qualidade de vinhos (Bom/Ruim) baseado em suas características físico-químicas usando Machine Learning com XGBoost e ADASYN.

## 🚀 Como Usar

### Opção 1: Modelo Pré-treinado (Recomendado)
O modelo já está treinado e pronto para uso! Basta executar:

```bash
streamlit run app.py
```

E navegar para a página "🔮 Predição" no menu lateral.

### Opção 2: Treinamento Automático
Se por algum motivo o modelo não estiver disponível, o sistema oferece treinamento automático:

1. Na página de predição, clique em "🤖 Treinar Modelo Automaticamente"
2. Aguarde o treinamento (pode levar alguns minutos)
3. O modelo será salvo automaticamente para uso futuro

### Opção 3: Treinamento Manual
Para treinar manualmente:

```bash
python create_pretrained_model.py
```

## 📊 Funcionalidades

### 1. Predição Individual
- Insira as características do vinho
- Receba predição (Bom/Ruim) com probabilidade
- Visualize explicações SHAP das predições

### 2. Análise em Lote
- Upload de arquivo CSV com múltiplos vinhos
- Predição em lote com download dos resultados

### 3. Explicações do Modelo
- Importância global das características
- Análise SHAP para entender as decisões do modelo

### 4. Exportar Modelo
- Exportar modelo treinado para uso em outros sistemas

## 🎯 Características do Modelo

- **Algoritmo**: XGBoost com ADASYN
- **Performance**: ~85% de acurácia
- **AUC-ROC**: ~0.87
- **Balanceamento**: ADASYN para lidar com classes desbalanceadas

## 📁 Estrutura de Arquivos

```
models/
├── xgboost_adasyn_model.pkl    # Modelo pré-treinado
pages/
├── 5_🔮_Predição.py            # Página de predição
create_pretrained_model.py      # Script para criar modelo
```

## 🔧 Dependências

Certifique-se de ter instalado:

```bash
pip install -r requirements.txt
```

## 💡 Dicas de Uso

1. **Primeira execução**: O modelo já está treinado, não precisa fazer nada
2. **Recarregar página**: Se houver problemas, recarregue a página do Streamlit
3. **Limpar cache**: Use `Ctrl+F5` para limpar cache se necessário
4. **SHAP**: Para explicações detalhadas, instale: `pip install shap`

## 🎉 Vantagens

✅ **Não precisa treinar sempre** - Modelo pré-treinado incluído  
✅ **Interface intuitiva** - Fácil de usar  
✅ **Explicações detalhadas** - SHAP para transparência  
✅ **Análise em lote** - Processamento de múltiplos vinhos  
✅ **Exportação** - Salvar resultados facilmente  

## 🚨 Solução de Problemas

### "Modelo não encontrado"
- Clique em "🤖 Treinar Modelo Automaticamente"
- Ou execute: `python create_pretrained_model.py`

### Erro de importação
- Instale dependências: `pip install -r requirements.txt`
- Para SHAP: `pip install shap`

### Erro de dados
- Verifique se `data/processed/wine-quality-combined.parquet` existe

## 📈 Métricas do Modelo

- **Acurácia**: ~85%
- **Precisão**: ~82%
- **Recall**: ~88%
- **F1-Score**: ~85%
- **AUC-ROC**: ~0.87

O modelo é otimizado para identificar vinhos de boa qualidade com alta confiabilidade. 