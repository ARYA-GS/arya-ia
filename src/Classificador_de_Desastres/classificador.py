# src/Classificador_de_Desastres/classificador.py

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# =======================
# 1. Carregamento de dados
# =======================

def carregar_dados(caminho_csv):
    """
    Carrega dados de um arquivo CSV para um DataFrame pandas.
    """
    try:
        df = pd.read_csv(caminho_csv)
        print(f"Dados carregados com sucesso de: {caminho_csv}")
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo não foi encontrado no caminho especificado: {caminho_csv}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro ao carregar os dados: {e}")
        return None

# =======================
# 2. Pré-processamento
# =======================

def preprocessar_dados(df, colunas_features, coluna_alvo):
    """
    Realiza o pré-processamento dos dados, incluindo Label Encoding para colunas
    categóricas e divisão em conjuntos de treino e teste.
    """
    df_encoded = df.copy()
    label_encoders = {}

    for col in colunas_features + [coluna_alvo]:
        if col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                # Garante que os valores são strings e remove espaços extras
                df_encoded[col] = df_encoded[col].astype(str).str.strip().fillna('VALOR_NULO').infer_objects(copy=False)
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                label_encoders[col] = le
            elif pd.api.types.is_numeric_dtype(df_encoded[col]):
                if df_encoded[col].isnull().sum() > 0:
                    df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mean())
            else:
                if df_encoded[col].isnull().sum() > 0:
                     df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mode()[0])

    X = df_encoded[colunas_features]
    y = df_encoded[coluna_alvo]

    if X.isnull().sum().sum() > 0:
        print("Aviso: Existem valores NaN em X. Preenchendo com a média/modo.")
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(X[col].mode()[0])

    if y.isnull().sum() > 0:
        print("Aviso: Existem valores NaN em y. Removendo linhas correspondentes.")
        valid_indices = y.dropna().index
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

    if X.empty or y.empty:
        raise ValueError("Após o pré-processamento e tratamento de NaNs, não há dados suficientes para treino.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, df_encoded, label_encoders

# =======================
# 3. Treinamento do Modelo
# =======================

def treinar_modelo(X_train, y_train):
    """
    Treina um modelo RandomForestClassifier.
    """
    print("Iniciando o treinamento do modelo RandomForestClassifier...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")
    return modelo

# =======================
# 4. Aplicar Previsões no DataFrame Original
# =======================

def aplicar_predicoes(df_original, modelo, colunas_features, label_encoder_alvo, label_encoders):
    """
    Aplica as previsões do modelo ao DataFrame original, decodificando
    a coluna alvo predita de volta para os rótulos originais.
    """
    df_copy = df_original.copy()

    for col in colunas_features:
        if col in df_copy.columns:
            if col in label_encoders:
                df_copy[col] = df_copy[col].astype(str).str.strip().fillna('VALOR_NULO').infer_objects(copy=False)
                df_copy[col] = label_encoders[col].transform(df_copy[col])
            elif pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_original[col].mean(numeric_only=True))
        else:
            print(f"Aviso: Coluna '{col}' não encontrada no DataFrame para aplicar predições.")
            df_copy[col] = 0

    X_pred = df_copy[colunas_features]

    predictions_encoded = modelo.predict(X_pred)

    df_copy['predicted_class'] = label_encoder_alvo.inverse_transform(predictions_encoded)

    return df_copy

# =======================
# 5. Criar GeoDataFrame
# =======================

def criar_geodataframe(df):
    """
    Cria um GeoDataFrame a partir de um DataFrame pandas, usando
    as colunas 'longitude' e 'latitude' para a geometria.
    """
    df_geo = df.dropna(subset=['latitude', 'longitude']).copy()

    df_geo['latitude'] = pd.to_numeric(df_geo['latitude'], errors='coerce')
    df_geo['longitude'] = pd.to_numeric(df_geo['longitude'], errors='coerce')
    df_geo.dropna(subset=['latitude', 'longitude'], inplace=True)

    if df_geo.empty:
        print("Aviso: Não há dados válidos de latitude/longitude para criar o GeoDataFrame.")
        return None

    geometry = [Point(xy) for xy in zip(df_geo['longitude'], df_geo['latitude'])]
    gdf = gpd.GeoDataFrame(df_geo, geometry=geometry)
    gdf.set_crs(epsg=4326, inplace=True)
    print("GeoDataFrame criado com sucesso!")
    return gdf

# =======================
# 6. Visualização com Plotly (Estilo Mapa de Calor com Bolhas)
# =======================
def visualizar_mapa_com_legenda(gdf):
    """
    Cria uma visualização de mapa interativa usando Plotly, mostrando
    a classificação de risco com base nas previsões do modelo.
    Retorna a figura Plotly.
    """
    if gdf is None or gdf.empty:
        print("Não há dados no GeoDataFrame para visualizar.")
        return None

    # NOVO: Ajuste do dicionário de cores para as classes reais do Classificador
    # BASEADO NO SEU CSV relatorios_desastres_20250608_141218.csv
    # Classes encontradas: 'Suporte Necessário', 'Risco Imediato', 'Monitorar', 'Atenção Urgente'
    cor_por_risco = {
        'Risco Imediato': 'darkred', # Mais crítico
        'Atenção Urgente': 'orange', # Atenção
        'Suporte Necessário': 'yellow', # Precisa de suporte, mas não crítico
        'Monitorar': 'blue', # Baixo risco, apenas monitoramento
        'DESCONHECIDO': 'gray' # Fallback para qualquer classe não mapeada
    }

    fig = go.Figure()

    for risco, grupo in gdf.groupby("predicted_class"):
        risco_limpo = str(risco).strip()
        cor = cor_por_risco.get(risco_limpo, 'gray')

        fig.add_trace(go.Scattermapbox(
            lat=grupo.geometry.y,
            lon=grupo.geometry.x,
            mode='markers',
            marker=dict(
                size=14,
                color=cor,
                opacity=0.8,
                symbol='circle'
            ),
            name=f"Risco: {risco_limpo}",
            hoverinfo='text',
            hovertext=grupo.apply(lambda row:
                f"<b>Classificação Prevista:</b> {row['predicted_class']}<br>"
                f"<b>Tipo de Observação:</b> {row.get('tipoObservacao', 'N/A')}<br>"
                f"<b>Nível de Gravidade:</b> {row.get('nivelGravidade', 'N/A')}<br>"
                f"<b>Qtd. Atingidos:</b> {row.get('qtdAtingidos', 'N/A')}<br>"
                f"<b>Dano Infraestrutura:</b> {row.get('danoInfraestrutura', 'N/A')}<br>"
                f"<b>Acessibilidade:</b> {row.get('acessibilidade', 'N/A')}<br>"
                f"<b>Notas:</b> {row.get('notasAdicionais', 'N/A')}", axis=1),
            showlegend=True
        ))

        fig.add_trace(go.Scattermapbox(
            lat=grupo.geometry.y,
            lon=grupo.geometry.x,
            mode='markers',
            marker=dict(
                size=80,
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
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title_text="Classificação de Risco de Desastres por Localização",
        title_x=0.5,
        legend=dict(
            title="Classes de Risco Previstas",
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='Black',
            borderwidth=1
        ),
    )
    return fig

# =======================
# 7. Execução principal (modificada para retornar a figura)
# =======================

def main_classificador():
    # Caminho do arquivo CSV. Ajustado para ser relativo ao diretório do script.
    caminho_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'CSV', 'relatorios_desastres_20250608_141218.csv')

    colunas_features = [
        'tipoFonte', 'tipoObservacao', 'nivelGravidade',
        'qtdAtingidos', 'danoInfraestrutura', 'acessibilidade'
    ]
    coluna_alvo = 'classificacaoRisco' # O alvo é 'classificacaoRisco'

    df = carregar_dados(caminho_csv)

    if df is not None:
        missing_features = [col for col in colunas_features if col not in df.columns]
        if missing_features:
            print(f"Erro: As seguintes colunas de features não foram encontradas no CSV: {missing_features}")
            print("Verifique se os nomes das colunas em 'colunas_features' e no seu CSV correspondem exatamente.")
            return None
        if coluna_alvo not in df.columns:
            print(f"Erro: A coluna alvo '{coluna_alvo}' não foi encontrada no CSV.")
            print("Verifique se o nome da coluna alvo no seu CSV corresponde exatamente.")
            return None

        # Conversão de tipo e imputação para colunas que vão para o modelo
        # Assegurar que 'qtdAtingidos' é numérica
        df['qtdAtingidos'] = pd.to_numeric(df['qtdAtingidos'], errors='coerce')
        df['qtdAtingidos'].fillna(df['qtdAtingidos'].mean(), inplace=True)

        # Imputação e strip() para as colunas que serão usadas como features e o alvo
        # Garante que colunas categóricas para o LabelEncoder sejam strings e limpas
        for col in colunas_features + [coluna_alvo]:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col].fillna('VALOR_NULO', inplace=True)


        X_train, X_test, y_train, y_test, df_encoded, label_encoders = preprocessar_dados(df.copy(), colunas_features, coluna_alvo)

        modelo = treinar_modelo(X_train, y_train)

        label_encoder_alvo = label_encoders[coluna_alvo]
        df_predito = aplicar_predicoes(df.copy(), modelo, colunas_features, label_encoder_alvo, label_encoders)
        
        # NOVO: Imputar NaNs e limpar strings nas colunas do df_predito que vão para o hovertext
        # Usar a mesma lógica de imputação das features do modelo
        hover_cols_to_impute = [
            'tipoObservacao', 'nivelGravidade', 'qtdAtingidos',
            'danoInfraestrutura', 'acessibilidade', 'notasAdicionais', # Estas são do hovertext
            'predicted_class' # A coluna que está no mapa
        ]
        for col_name in hover_cols_to_impute:
            if col_name in df_predito.columns:
                if pd.api.types.is_numeric_dtype(df_predito[col_name]):
                    df_predito[col_name].fillna(df_predito[col_name].mean(), inplace=True)
                elif df_predito[col_name].dtype == 'object' or pd.api.types.is_categorical_dtype(df_predito[col_name]):
                    df_predito[col_name] = df_predito[col_name].astype(str).str.strip().fillna('N/A')
                    df_predito[col_name] = df_predito[col_name].replace('nan', 'N/A')
            else:
                # Se uma coluna esperada no hovertext não existe no df_predito, adiciona como N/A
                df_predito[col_name] = 'N/A'


        gdf = criar_geodataframe(df_predito)

        if gdf is not None:
            return visualizar_mapa_com_legenda(gdf)
        else:
            print("Não foi possível gerar o GeoDataFrame.")
            return None
    else:
        print("Não foi possível carregar os dados. O programa será encerrado.")
        return None