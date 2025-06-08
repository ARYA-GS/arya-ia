# src/Classificador_de_Desastres/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import os

def run_classification():
    """
    Executa o modelo de classificação de desastres.
    Lê os dados, treina um RandomForestClassifier, faz previsões e
    gera um mapa Plotly com a classificação de risco.
    Retorna:
        go.Figure: A figura do mapa Plotly.
    """
    # Garante que o caminho para o arquivo de dados esteja correto
    # assumindo que o script é executado a partir da raiz do projeto (onde está o app.py)
    file_path = "C:/Users/zenet/OneDrive/Desktop/GS_IA_FINAL/arya-ia/Data/relatorios_desastres.csv"
    
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        return go.Figure().update_layout(title_text=f"Erro: Arquivo não encontrado em {file_path}")

    # Pré-processamento dos dados
    data = data.dropna(subset=['latitude', 'longitude', 'AREA_RISK_CLASSIFICATION'])
    data.fillna('Não Informado', inplace=True)

    categorical_cols = [
        'source_type', 'observation_type', 'severity_reported', 
        'infrastructure_damage', 'accessibility'
    ]
    
    label_encoders = {}
    for column in categorical_cols:
        if column in data.columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le
        else:
             return go.Figure().update_layout(title_text=f"Erro: Coluna categórica '{column}' não encontrada no CSV.")


    features = [
        'source_type', 'observation_type', 'severity_reported', 
        'num_affected_estimate', 'infrastructure_damage', 'accessibility'
    ]
    
    # Verifica se todas as features necessárias existem no DataFrame
    for feature in features:
        if feature not in data.columns:
             return go.Figure().update_layout(title_text=f"Erro: Coluna de feature '{feature}' não encontrada no CSV.")

    X = data[features]
    y = data['AREA_RISK_CLASSIFICATION']

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinamento do modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # Avaliação
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do Classificador de Desastres: {accuracy:.2f}")

    # Previsão para todos os dados para visualização
    data['predicted_risk'] = model.predict(X)

    # Criação do mapa
    color_map = {
        'Low': '#2ca02c',       # Verde
        'Medium': '#ff7f0e',    # Laranja
        'High': '#d62728',      # Vermelho
        'Critical': '#8c564b'   # Marrom (vermelho escuro)
    }

    fig = go.Figure()

    for risk, color in color_map.items():
        df_risk = data[data['predicted_risk'] == risk]
        if not df_risk.empty:
            fig.add_trace(go.Scattermapbox(
                lat=df_risk['latitude'],
                lon=df_risk['longitude'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color=color,
                    opacity=0.7
                ),
                text=[f"Tipo: {obs}<br>Gravidade: {sev}" for obs, sev in zip(df_risk['observation_type'].map(lambda x: label_encoders['observation_type'].inverse_transform([x])[0]), df_risk['severity_reported'].map(lambda x: label_encoders['severity_reported'].inverse_transform([x])[0]))],
                hoverinfo='text',
                name=risk
            ))

    fig.update_layout(
        title_text='<b>Classificação de Risco de Desastres Reportados no Brasil</b>',
        title_x=0.5,
        mapbox_style="open-street-map",
        mapbox_center_lon=-55,
        mapbox_center_lat=-14,
        mapbox_zoom=3.5,
        legend_title_text='Nível de Risco Previsto',
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    # MODIFICAÇÃO PRINCIPAL: Retorna a figura para o Flask
    return fig

# A chamada principal foi removida para não auto-executar quando importado
# if __name__ == '__main__':
#     fig = run_classification()
#     fig.show()