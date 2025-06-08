# desastres_modelo_visualizacao.py

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

    # Tratar colunas que podem ter valores nulos ou tipos mistos antes do encoding
    for col in colunas_features + [coluna_alvo]:
        if col in df_encoded.columns:
            # Para colunas categóricas (object), converter para string e preencher NaNs
            if df_encoded[col].dtype == 'object':
                df_encoded[col] = df_encoded[col].astype(str).fillna('VALOR_NULO').infer_objects(copy=False)
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                label_encoders[col] = le
            # Para colunas numéricas, preencher NaNs com a média
            elif pd.api.types.is_numeric_dtype(df_encoded[col]):
                if df_encoded[col].isnull().sum() > 0:
                    df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mean())
            # Se for dataHora, tentar converter para timestamp numérico ou similar, ou extrair features
            # Por simplicidade, para este modelo, não usaremos dataHora como feature diretamente
            # mas podemos discutir isso como uma melhoria futura.

    X = df_encoded[colunas_features]
    y = df_encoded[coluna_alvo]

    # Garantir que não há NaNs nas features ou no alvo antes de treinar
    # Isso é crucial para muitos modelos do sklearn.
    if X.isnull().sum().sum() > 0:
        print("Aviso: Existem valores NaN em X. Preenchendo com a média/modo.")
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else: # Colunas categóricas já deveriam estar numéricas após LabelEncoder
                # Se ainda houver NaNs aqui, é um problema de processamento anterior
                # ou tipo de dado não tratado. Preenchendo com o modo para segurança.
                X[col] = X[col].fillna(X[col].mode()[0])

    if y.isnull().sum() > 0:
        print("Aviso: Existem valores NaN em y. Removendo linhas correspondentes.")
        # Remover linhas onde o alvo é NaN é a abordagem mais segura para classificação
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

    # Preprocessa as colunas de features no df_copy usando os mesmos encoders
    for col in colunas_features:
        if col in df_copy.columns:
            if col in label_encoders: # Se a coluna foi LabelEncoded no treino
                # Certifica que os dados de inferência são strings e trata NaNs
                df_copy[col] = df_copy[col].astype(str).fillna('VALOR_NULO').infer_objects(copy=False)
                # Tenta transformar. Se um valor não foi visto no treino, o transform pode falhar.
                # Para robustez, podemos mapear para um valor padrão ou lidar com KeyError.
                # Por agora, assumimos que os valores de inferência são um subconjunto dos de treino.
                df_copy[col] = label_encoders[col].transform(df_copy[col])
            elif pd.api.types.is_numeric_dtype(df_copy[col]): # Se for numérica e não foi encoded
                # Preencher NaNs com a média da coluna (idealmente, a média do *treino*)
                df_copy[col] = df_copy[col].fillna(df_original[col].mean(numeric_only=True))
        else:
            # Se a coluna não existe no df_copy, mas está em colunas_features, isso é um problema.
            # Idealmente, isso deveria ser pego por uma verificação antes.
            print(f"Aviso: Coluna '{col}' não encontrada no DataFrame para aplicar predições.")
            df_copy[col] = 0 # Preenche com um valor padrão para evitar erro no X_pred


    X_pred = df_copy[colunas_features]

    # Realiza a previsão
    predictions_encoded = modelo.predict(X_pred)

    # Decodifica as previsões de volta para os rótulos originais
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
    # Remover linhas com valores nulos em 'latitude' ou 'longitude' para evitar erros
    df_geo = df.dropna(subset=['latitude', 'longitude']).copy()

    # Garantir que latitude e longitude são numéricas
    df_geo['latitude'] = pd.to_numeric(df_geo['latitude'], errors='coerce')
    df_geo['longitude'] = pd.to_numeric(df_geo['longitude'], errors='coerce')
    df_geo.dropna(subset=['latitude', 'longitude'], inplace=True) # Remover NaNs após coerção

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
    """
    if gdf is None or gdf.empty:
        print("Não há dados no GeoDataFrame para visualizar.")
        return

    # Mapeamento de cores para as classes de risco (ajustado para os rótulos do seu dataset)
    # Adapte estas cores e classes conforme as classes reais que 'classificacaoRisco' pode ter.
    cor_por_risco = {
        'ALTO': 'red',
        'MEDIO': 'orange',
        'BAIXO': 'green',
        'CRITICO': 'darkred', # Exemplo de nova categoria
        'MODERADO': 'yellow', # Exemplo de nova categoria
        'DESCONHECIDO': 'gray' # Para qualquer classe não mapeada
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
                opacity=0.8,
                symbol='circle'
            ),
            name=f"Risco: {risco}",
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

        # Círculo de impacto (visual auxiliar)
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
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title_text="Classificação de Risco de Desastres por Localização",
        title_x=0.5,
        legend=dict(
            title="Classes de Risco Previstas",
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='Black',
            borderwidth=1
        )
    )
    print("Mapa gerado. Abrindo visualização...")
    fig.show()

# =======================
# 7. Execução principal
# =======================

def main():
    # Caminho do arquivo CSV (ajuste conforme a localização real do seu arquivo)
    # Por exemplo, se o arquivo estiver na mesma pasta do script, use apenas "seuarquivo.csv"
    caminho_csv = "C:/Users/zenet/OneDrive/Desktop/ARYA_IA_GS/arya-ia/Data/CSV/relatorios_desastres_20250608_141218.csv"

    # NOVAS COLUNAS DE FEATURES E COLUNA ALVO BASEADAS NO DATASET CORRETO
    colunas_features = [
        'tipoFonte', 'tipoObservacao', 'nivelGravidade',
        'qtdAtingidos', 'danoInfraestrutura', 'acessibilidade'
        # 'dataHora' e 'notasAdicionais' são ignoradas por simplicidade.
        # 'dataHora' poderia ser usada para extrair features de tempo.
        # 'notasAdicionais' demandaria Processamento de Linguagem Natural.
    ]
    coluna_alvo = 'classificacaoRisco'

    df = carregar_dados(caminho_csv)

    if df is not None:
        # Verifica se todas as colunas necessárias existem no DataFrame
        missing_features = [col for col in colunas_features if col not in df.columns]
        if missing_features:
            print(f"Erro: As seguintes colunas de features não foram encontradas no CSV: {missing_features}")
            print("Verifique se os nomes das colunas em 'colunas_features' e no seu CSV correspondem exatamente.")
            return
        if coluna_alvo not in df.columns:
            print(f"Erro: A coluna alvo '{coluna_alvo}' não foi encontrada no CSV.")
            print("Verifique se o nome da coluna alvo no seu CSV corresponde exatamente.")
            return

        # Converter colunas numéricas (se houver alguma que o pandas não detectou automaticamente)
        # O dataset atual tem 'qtdAtingidos'. Garantir que seja numérica.
        df['qtdAtingidos'] = pd.to_numeric(df['qtdAtingidos'], errors='coerce')


        X_train, X_test, y_train, y_test, df_encoded, label_encoders = preprocessar_dados(df.copy(), colunas_features, coluna_alvo)

        modelo = treinar_modelo(X_train, y_train)

        label_encoder_alvo = label_encoders[coluna_alvo]
        df_predito = aplicar_predicoes(df.copy(), modelo, colunas_features, label_encoder_alvo, label_encoders)

        gdf = criar_geodataframe(df_predito)

        if gdf is not None:
            visualizar_mapa_com_legenda(gdf)
        else:
            print("Não foi possível gerar o GeoDataFrame, a visualização não será exibida.")
    else:
        print("Não foi possível carregar os dados. O programa será encerrado.")


if __name__ == "__main__":
    main()