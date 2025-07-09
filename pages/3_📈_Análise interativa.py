# Teste de Random Forest com regularização mais forte
rf_reg = RandomForestClassifier(
    n_estimators=grid_search_rf.best_params_['n_estimators'],
    max_depth=grid_search_rf.best_params_['max_depth'],
    min_samples_split=5,   # Aumente para 5 ou 10
    min_samples_leaf=3,    # Aumente para 3 ou 5
    max_features=grid_search_rf.best_params_['max_features'],
    class_weight=grid_search_rf.best_params_['class_weight'],
    criterion=grid_search_rf.best_params_['criterion'],
    random_state=42
)
rf_reg.fit(X_train_resampled, y_train_resampled)
print('Modelo Random Forest regularizado treinado!') 