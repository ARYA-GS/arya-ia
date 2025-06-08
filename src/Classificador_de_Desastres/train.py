import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

def carregar_dados(caminho_csv):
    script_dir = os.path.dirname(__file__)
    # Caminho corrigido para buscar da pasta /Data na raiz do projeto
    caminho_abs = os.path.join(script_dir, '..', '..', 'Data', caminho_csv)
    df = pd.read_csv(caminho_abs)
    return df

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
    return X, y, df_encoded, label_encoders

def treinar_modelo(X_train, y_train):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def aplicar_predicoes(df_original, modelo, colunas_features, label_encoder_alvo, label_encoders):
    df_copy = df_original.copy()
    X_pred_encoded = df_copy.copy()
    for col in colunas_features:
        if col in label_encoders:
            X_pred_encoded[col] = label_encoders[col].transform(df_copy[col])
    X_pred = X_pred_encoded[colunas_features]
    df_copy['predicted_class'] = label_encoder_alvo.inverse_transform(modelo.predict(X_pred))
    return df_copy

def criar_geodataframe(df):
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf

def visualizar_mapa_com_legenda(gdf):
    cor_por_risco = {'Risco Imediato': 'red', 'Atenção Urgente': 'orange', 'Suporte Necessário': 'yellow', 'Monitorar': 'blue'}
    fig = go.Figure()
    for risco, grupo in gdf.groupby("predicted_class"):
        cor = cor_por_risco.get(risco, 'gray')
        hover_texts = grupo.apply(lambda row: f"<b>Classificação:</b> {row['predicted_class']}<br><b>Tipo:</b> {row.get('tipoObservacao', 'N/A')}<br><b>Severidade:</b> {row.get('nivelGravidade', 'N/A')}<br><b>Atingidos:</b> {row.get('pessoasAfetadas', 'N/A')}<br><b>Notas:</b> {row.get('detalhesAdicionais', 'N/A')}", axis=1)
        fig.add_trace(go.Scattermapbox(lat=grupo.geometry.y, lon=grupo.geometry.x, mode='markers', marker=dict(size=14, color=cor, opacity=0.8), name=risco, hoverinfo='text', hovertext=hover_texts))
        fig.add_trace(go.Scattermapbox(lat=grupo.geometry.y, lon=grupo.geometry.x, mode='markers', marker=dict(size=80, color=cor, opacity=0.2), hoverinfo='skip', showlegend=False))
    fig.update_layout(mapbox=dict(style="open-street-map", zoom=3.5, center=dict(lat=gdf.geometry.y.mean(), lon=gdf.geometry.x.mean())), margin={"r":0,"t":40,"l":0,"b":0}, title="Classificação de Risco de Desastres", legend=dict(title="Classes de Risco", x=0, y=1, bgcolor='rgba(255, 255, 255, 0.7)'))
    return fig

def run():
    caminho_csv = "relatorios_desastres.csv"
    colunas_features = ['fonte', 'tipoObservacao', 'nivelGravidade', 'pessoasAfetadas', 'danosInfraestrutura', 'acessibilidadeLocal']
    coluna_alvo = 'classificacaoRisco'
    df = carregar_dados(caminho_csv)
    X, y, df_encoded, label_encoders = preprocessar_dados(df, colunas_features, coluna_alvo)
    modelo = treinar_modelo(X, y)
    df_predito = aplicar_predicoes(df, modelo, colunas_features, label_encoders[coluna_alvo], label_encoders)
    gdf = criar_geodataframe(df_predito)
    return visualizar_mapa_com_legenda(gdf)

if __name__ == "__main__":
    figura_gerada = run()
    figura_gerada.show()