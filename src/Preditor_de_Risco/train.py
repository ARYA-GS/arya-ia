# src/Preditor_de_Risco/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import os

def run_prediction():
    """
    Executa o modelo de predição de ocorrência de eventos.
    Lê os dados, treina um modelo LightGBM, calcula a probabilidade de risco
    e gera um mapa de calor Plotly.
    Retorna:
        go.Figure: A figura do mapa Plotly.
        float: A acurácia do modelo treinado.
    """
    file_path = file_path = "C:/Users/zenet/OneDrive/Desktop/GS_IA_FINAL/arya-ia/Data/desastres_naturais.csv"
    
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        fig = go.Figure().update_layout(title_text=f"Erro: Arquivo não encontrado em {file_path}")
        return fig, 0.0

    # Pré-processamento
    data = data.dropna(subset=['latitude', 'longitude', 'ocorrencia_evento'])
    data.fillna(data.median(numeric_only=True), inplace=True)
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    if 'ocorrencia_evento' not in data.columns:
        fig = go.Figure().update_layout(title_text="Erro: Coluna alvo 'ocorrencia_evento' não encontrada.")
        return fig, 0.0

    X = data.drop('ocorrencia_evento', axis=1)
    y = data['ocorrencia_evento']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.drop(['latitude', 'longitude'], errors='ignore')
    categorical_features = X.select_dtypes(include=['object']).columns

    # Criação do preprocessor para o pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Mantém colunas não transformadas (como lat/lon)
    )

    # Definição do pipeline com o melhor modelo (LightGBM)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Treinamento
    pipeline.fit(X_train, y_train)
    
    # Avaliação
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do Preditor de Risco (LightGBM): {accuracy:.2f}")

    # Previsão da probabilidade para todos os dados para visualização
    probabilities = pipeline.predict_proba(X)[:, 1]
    data['risk_probability'] = probabilities

    # Criação do mapa de calor
    fig = go.Figure(go.Scattermapbox(
        lat=data['latitude'],
        lon=data['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            color=data['risk_probability'],
            colorscale='YlOrRd',
            cmin=0,
            cmax=1,
            colorbar_title="Probabilidade<br>de Risco"
        ),
        text=data.apply(lambda row: f"ID da Zona: {row.get('id_zona', 'N/A')} <br>Probabilidade: {row['risk_probability']:.2%}", axis=1),
        hoverinfo='text'
    ))

    fig.update_layout(
        title_text='<b>Predição de Probabilidade de Risco de Desastres no Brasil</b>',
        title_x=0.5,
        mapbox_style="open-street-map",
        mapbox_center_lon=-55,
        mapbox_center_lat=-14,
        mapbox_zoom=3.5,
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    # MODIFICAÇÃO PRINCIPAL: Retorna a figura e a acurácia para o Flask
    return fig, accuracy

# A chamada principal foi removida para não auto-executar quando importado
# if __name__ == '__main__':
#     fig, acc = run_prediction()
#     print(f"Acurácia final: {acc}")
#     fig.show()