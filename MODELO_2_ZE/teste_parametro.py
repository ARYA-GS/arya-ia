import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

# Seed para reproducibilidade
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 1. Carregar dataset
df = pd.read_csv('C:/Users/zenet/OneDrive/Desktop/GS_IA_FINAL/arya-ia/MODELO_2_ZE/dataset_risco_desastres_v2.csv')

# 2. Pré-processamento
df.drop(['id_zona', 'data_referencia', 'tipo_evento'], axis=1, inplace=True)

# Variáveis categóricas
categoricas_features = ['tipo_solo', 'uso_solo', 'acessibilidade', 'estacao']
coluna_alvo = 'risco_previsto'

# Converter colunas categóricas para o tipo category
for col in categoricas_features:
    if col in df.columns:
        df[col] = df[col].astype('category')
    else:
        print(f"Atenção: Coluna '{col}' não encontrada para conversão categórica.")

# Codifica variável alvo
le_dict = {}
if coluna_alvo in df.columns:
    le_target = LabelEncoder()
    df[coluna_alvo] = le_target.fit_transform(df[coluna_alvo].astype(str))
    le_dict[coluna_alvo] = le_target
else:
    raise KeyError(f"Coluna alvo '{coluna_alvo}' não encontrada no DataFrame.")

# 3. Definir X e y
X = df.drop(coluna_alvo, axis=1)
y = df[coluna_alvo]

# 4. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# 5. Espaço de busca de hiperparâmetros
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 6, 8, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'min_child_weight': [1, 3, 5]
}

# 6. Modelo base
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    enable_categorical=True,
    random_state=SEED
)

# 7. Busca com RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=30,
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=SEED,
    n_jobs=-1
)

print("\n🔍 Iniciando busca de hiperparâmetros...")
random_search.fit(X_train, y_train)

# 8. Melhores parâmetros encontrados
print("\n🎯 Melhores hiperparâmetros encontrados:")
print(random_search.best_params_)

# 9. Avaliação com o melhor modelo
best_xgb = random_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)

# 10. Relatório de classificação
target_names_display = le_dict[coluna_alvo].classes_

print("\n📊 Relatório de Classificação com melhores hiperparâmetros:\n")
print(classification_report(y_test, y_pred_best, target_names=target_names_display))

# 11. Matriz de confusão
plt.figure(figsize=(7, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues',
    xticklabels=target_names_display,
    yticklabels=target_names_display
)
plt.title('Matriz de Confusão - XGBoost com Hiperparâmetros Otimizados')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.tight_layout()
plt.show()
