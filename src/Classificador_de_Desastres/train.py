# desastres_modelo_visualizacao.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# =======================
# 1. Carregamento de dados
# =======================

def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    return df

# =======================
# 2. Pré-processamento
# =======================

def preprocessar_dados(df, colunas_features, coluna_alvo):
    df_encoded = df.copy()
    label_encoders = {}

    for col in colunas_features + [coluna_alvo]:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

    X = df_encoded[colunas_features]
    y = df_encoded[coluna_alvo]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, df_encoded, label_encoders

# =======================
# 3. Treinamento do Modelo
# =======================

def treinar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

# =======================
# 4. Aplicar Previsões no DataFrame Original
# =======================

def aplicar_predicoes(df_original, modelo, colunas_features, label_encoder_alvo, label_encoders):
    df_copy = df_original.copy()

    for col in colunas_features:
        if col in label_encoders:
            df_copy[col] = label_encoders[col].transform(df_copy[col])

    X_pred = df_copy[colunas_features]
    df_copy['predicted_class'] = label_encoder_alvo.inverse_transform(modelo.predict(X_pred))

    return df_copy

# =======================
# 5. Criar GeoDataFrame
# =======================

def criar_geodataframe(df):
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf

# =======================
# 6. Visualização com Plotly (mapa estilo Google Maps)
# =======================

# =======================
# 6. Visualização com Plotly (Estilo Mapa de Calor com Bolhas)
# =======================

def visualizar_mapa_com_legenda(gdf):
    # Cores fixas por classe de risco
    cor_por_risco = {
        'Risco Imediato': 'red',
        'Atenção Urgente': 'orange',
        'Suporte Necessário': 'yellow',
        'Monitorar': 'blue'
    }

    fig = go.Figure()

    for risco, grupo in gdf.groupby("predicted_class"):
        cor = cor_por_risco.get(risco, 'gray')

        # Marcadores principais (pontos com hover)
        fig.add_trace(go.Scattermapbox(
            lat=grupo.geometry.y,
            lon=grupo.geometry.x,
            mode='markers',
            marker=dict(
                size=14,
                color=cor,
                opacity=0.8
            ),
            name=risco,
            hoverinfo='text',
            hovertext=grupo.apply(lambda row:
                f"<b>Classificação:</b> {row['predicted_class']}<br>"
                f"<b>Tipo:</b> {row.get('observation_type', '')}<br>"
                f"<b>Severidade:</b> {row.get('severity_reported', '')}<br>"
                f"<b>Atingidos:</b> {row.get('num_affected_estimate', '')}<br>"
                f"<b>Notas:</b> {row.get('additional_notes', '')}", axis=1)
        ))

        # Círculo de impacto (só visual)
        fig.add_trace(go.Scattermapbox(
            lat=grupo.geometry.y,
            lon=grupo.geometry.x,
            mode='markers',
            marker=dict(
                size=80,  # aumentar para simular raio de alcance maior
                color=cor,
                opacity=0.2
            ),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=5,
            center=dict(lat=gdf.geometry.y.mean(), lon=gdf.geometry.x.mean())
        ),
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title="Classificação de Risco com Áreas de Impacto Ampliadas",
        legend=dict(title="Classes de Risco", x=0, y=1)
    )

    fig.show()



# =======================
# 7. Execução principal
# =======================

def main():
    caminho_csv = "C:/Users/zenet/OneDrive/Desktop/GS_IA_FINAL/arya-ia/MODELO_1_VITOR/DATA/desastres_reports_brasil.csv"

    colunas_features = [
        'source_type', 'observation_type', 'severity_reported',
        'num_affected_estimate', 'infrastructure_damage', 'accessibility'
    ]
    coluna_alvo = 'AREA_RISK_CLASSIFICATION'

    df = carregar_dados(caminho_csv)
    X_train, X_test, y_train, y_test, df_encoded, label_encoders = preprocessar_dados(df, colunas_features, coluna_alvo)
    modelo = treinar_modelo(X_train, y_train)
    df_predito = aplicar_predicoes(df, modelo, colunas_features, label_encoders[coluna_alvo], label_encoders)
    gdf = criar_geodataframe(df_predito)
    visualizar_mapa_com_legenda(gdf)

if __name__ == "__main__":
    main()
