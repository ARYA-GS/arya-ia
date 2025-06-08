import pandas as pd
import numpy as np
import time
import geopandas
import plotly.graph_objects as go

# Preprocessing and Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Metrics
from sklearn.metrics import (
    accuracy_score,
    # precision_score, # Removido se não usado diretamente
    # recall_score, # Removido se não usado diretamente
    # f1_score, # Removido se não usado diretamente
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc
)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # print("Biblioteca SHAP não instalada. A interpretabilidade com SHAP será pulada.") # Comentado para reduzir output

# --- Configurações Globais ---
RANDOM_STATE = 42
TARGET_COLUMN = 'ocorrencia_evento'
ENABLE_STATIC_PLOTS = False # NOVO: Flag para controlar plots estáticos (feature importance, SHAP)

# --- 1. Carregamento e Preparação Inicial dos Dados ---
def load_and_prepare_data(filepath='C:/Users/zenet/OneDrive/Desktop/GS_IA_FINAL/arya-ia/MODELO_2_ZE/dataset_risco_desastres_brasil_v2.csv'):
    """Carrega os dados e faz uma preparação inicial."""
    try:
        df_full = pd.read_csv(filepath) # MODIFICADO: Carrega o DataFrame completo primeiro
    except FileNotFoundError:
        print(f"Arquivo '{filepath}' não encontrado. Por favor, gere-o primeiro.")
        raise
    
    print("Dados carregados com sucesso.")
    print(f"Shape inicial: {df_full.shape}")

    # Colunas para o modelo (X_model) e para o mapa (X_map_info)
    # TARGET_COLUMN é 'ocorrencia_evento'
    # 'risco_previsto' é derivado das regras, não usar no modelo.
    # 'id_zona', 'data_referencia' são identificadores.
    
    # Features para o modelo: não deve incluir 'tipo_evento' se for leakage
    cols_to_drop_for_model = ['id_zona', 'data_referencia', 'tipo_evento', 'risco_previsto', TARGET_COLUMN]
    
    # Certificar que as colunas a serem dropadas existem
    cols_present_to_drop_for_model = [col for col in cols_to_drop_for_model if col in df_full.columns]
    
    X_model_df = df_full.drop(columns=cols_present_to_drop_for_model, errors='ignore')
    y_series = df_full[TARGET_COLUMN]
    
    # Informações adicionais para o mapa (incluindo aquelas que podem ser usadas no hover)
    # Manter o índice original para garantir o alinhamento após o train_test_split
    map_info_cols = ['lat', 'lon', 'tipo_evento', 'chuva_mm', 'temperatura_media', 'uso_solo']
    # Garantir que apenas colunas existentes são selecionadas
    map_info_cols_present = [col for col in map_info_cols if col in df_full.columns]
    X_map_info_df = df_full[map_info_cols_present].copy() # Usar .copy() para evitar SettingWithCopyWarning

    print(f"Shape de X_model_df (features para o modelo): {X_model_df.shape}")
    print(f"Shape de X_map_info_df (infos para o mapa): {X_map_info_df.shape}")
    return X_model_df, y_series, X_map_info_df


def identify_feature_types(df_for_model_features): # MODIFICADO: Recebe apenas features do modelo
    """Identifica features numéricas e categóricas para o modelo."""
    # X é o DataFrame de features do modelo
    numerical_features = df_for_model_features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df_for_model_features.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Features numéricas (modelo): {numerical_features}")
    print(f"Features categóricas (modelo): {categorical_features}")
    return numerical_features, categorical_features

# --- 2. Pré-processamento (igual) ---
def get_preprocessor(numerical_features, categorical_features):
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

# --- 3. Treinamento e Avaliação (igual) ---
def evaluate_model(y_true, y_pred, y_pred_proba, model_name="Model"):
    print(f"\n--- Resultados para {model_name} ---")
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    if len(np.unique(y_true)) > 1:
        print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        print(f"PR AUC: {pr_auc:.4f}")
    else:
        print("ROC AUC e PR AUC não podem ser calculados.")
    print("\nMatriz de Confusão:"); print(confusion_matrix(y_true, y_pred))
    print("\nRelatório de Classificação:")
    target_names_report = ['Não Evento (0)', 'Evento (1)'] if len(np.unique(y_true)) > 1 else [f'Classe {np.unique(y_true)[0]}']
    print(classification_report(y_true, y_pred, target_names=target_names_report, zero_division=0))

# --- 4. Interpretabilidade ---
# (Funções get_feature_names_from_preprocessor, display_feature_importance, explain_with_shap permanecem, mas sua chamada será condicional)
def get_feature_names_from_preprocessor(preprocessor, X_cols_original_names=None):
    try: return preprocessor.get_feature_names_out()
    except AttributeError:
        # ... (lógica de fallback como antes, pode ser simplificada se não for mais usada extensivamente) ...
        print("Aviso: preprocessor.get_feature_names_out() não disponível ou falhou. Retornando nomes genéricos.")
        # Se X_cols_original_names for o DataFrame original de features do modelo:
        if X_cols_original_names is not None:
             num_cols_processed = 0
             try: # Tenta transformar um DataFrame vazio para inferir o número de colunas de saída
                 num_cols_processed = preprocessor.transform(pd.DataFrame(columns=X_cols_original_names.columns)).shape[1]
             except Exception: # Se falhar, tenta com um array de zeros
                 num_cols_processed = preprocessor.transform(np.zeros((1, len(X_cols_original_names.columns)))).shape[1]
             return [f"feature_{i}" for i in range(num_cols_processed)]
        return ["feature_unknown"]


def display_feature_importance(pipeline, X_cols_original_model_features, model_name="Model"):
    if not ENABLE_STATIC_PLOTS: return # NOVO
    # ... (resto da função como antes) ...
    plt.show(block=False)

def explain_with_shap(pipeline, X_test_model_features, model_name="Model"):
    if not ENABLE_STATIC_PLOTS or not SHAP_AVAILABLE: return # NOVO
    # ... (resto da função como antes, usando X_test_model_features) ...
    plt.show(block=False)
    plt.show(block=False) # Para os dois plots SHAP


# --- NOVA Seção: Visualização do Mapa (Ajustada) ---
# --- NOVA Seção: Visualização do Mapa (Ajustada) ---
# --- NOVA Seção: Visualização do Mapa (Ajustada) ---
# --- NOVA Seção: Visualização do Mapa (Ajustada) ---
# --- NOVA Seção: Visualização do Mapa (Ajustada) ---
# --- Visualização do Mapa (MODIFICADA para Melhor Estética e Correção de Erro) ---
# --- Visualização do Mapa (MODIFICADA para Melhor Estética e Estilo de Mapa) ---
def display_classification_map(df_map_with_preds_and_info, lat_col='lat', lon_col='lon',
                               proba_col='probability', event_type_col='tipo_evento',
                               title="Mapa de Classificação de Risco"):
    """
    Exibe um mapa interativo com pontos sólidos e halos transparentes (estética aprimorada).
    """
    print(f"Gerando mapa com Plotly Graph Objects. Amostra dos dados (antes de filtrar 'nenhum'):\n{df_map_with_preds_and_info.head(3)}")

    df_filtered = df_map_with_preds_and_info[df_map_with_preds_and_info[event_type_col] != 'nenhum'].copy()
    if df_filtered.empty:
        print("Nenhum evento para exibir no mapa após filtrar 'nenhum'.")
        return

    # 1. Definir Nivel_Risco (severidade)
    bins_risco = [-0.01, 0.3, 0.7, 1.01]
    labels_risco = ['Baixo Risco', 'Médio Risco', 'Alto Risco']
    df_filtered['Nivel_Risco_Predito'] = pd.cut(
        df_filtered[proba_col], bins=bins_risco, labels=labels_risco, right=True, include_lowest=True
    )

    # 2. Definir cores BASE para event_type_col
    base_event_colors = {
        'inundação': 'blue', 
        'deslizamento': 'saddlebrown',
        'seca': 'orange', 
        'incêndio': 'red', 
        'vendaval': 'darkviolet' 
    }
    df_filtered['base_color'] = df_filtered[event_type_col].map(lambda x: base_event_colors.get(x, 'grey'))

    # 3. Definir tamanhos para pontos sólidos e halos
    solid_point_size_map = {'Baixo Risco': 8, 'Médio Risco': 12, 'Alto Risco': 16}
    halo_effect_size_map = {'Baixo Risco': 30, 'Médio Risco': 45, 'Alto Risco': 60}
    
    df_filtered['solid_marker_size'] = df_filtered['Nivel_Risco_Predito'].map(solid_point_size_map).astype(float).fillna(7.0).astype(int)
    df_filtered['halo_marker_size'] = df_filtered['Nivel_Risco_Predito'].map(halo_effect_size_map).astype(float).fillna(20.0).astype(int)

    # 4. Criar hovertext para os pontos sólidos
    hover_texts = []
    for _, row in df_filtered.iterrows():
        text = (f"<b>Tipo Evento:</b> {row[event_type_col]}<br>"
                f"<b>Nível Risco:</b> {row['Nivel_Risco_Predito']}<br>"
                f"<b>Probabilidade:</b> {row[proba_col]:.3f}<br>"
                f"Lat: {row[lat_col]:.4f}, Lon: {row[lon_col]:.4f}<br>")
        if 'chuva_mm' in row and pd.notna(row['chuva_mm']): text += f"Chuva (mm): {row['chuva_mm']}<br>"
        if 'temperatura_media' in row and pd.notna(row['temperatura_media']): text += f"Temp. Média: {row['temperatura_media']}<br>"
        if 'uso_solo' in row and pd.notna(row['uso_solo']): text += f"Uso do Solo: {row['uso_solo']}"
        hover_texts.append(text)
    df_filtered['hover_text'] = hover_texts

    # 5. Criar a figura com plotly.graph_objects
    fig = go.Figure()

    # Camada 1: Halos
    fig.add_trace(go.Scattermapbox(
        lat=df_filtered[lat_col],
        lon=df_filtered[lon_col],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=df_filtered['halo_marker_size'],
            color=df_filtered['base_color'], 
            opacity=0.12
        ),
        hoverinfo='skip',
        name='Área de Influência (Halo)',
        showlegend=False 
    ))

    # Camada 2: Pontos Sólidos
    legend_names_added = set()
    sorted_event_types = sorted(df_filtered[event_type_col].unique())

    for event_type_val in sorted_event_types:
        df_subset = df_filtered[df_filtered[event_type_col] == event_type_val]
        if df_subset.empty:
            continue
        show_legend_for_trace = event_type_val not in legend_names_added
        
        fig.add_trace(go.Scattermapbox(
            lat=df_subset[lat_col],
            lon=df_subset[lon_col],
            mode='markers', 
            marker=go.scattermapbox.Marker(
                size=df_subset['solid_marker_size'],
                color=df_subset['base_color'],
                opacity=0.9
            ),
            hovertext=df_subset['hover_text'],
            hoverinfo='text',
            name=str(event_type_val).capitalize(),
            showlegend=show_legend_for_trace
        ))
        if show_legend_for_trace:
            legend_names_added.add(event_type_val)

    map_center_lat = df_filtered[lat_col].mean() if not df_filtered[lat_col].empty else -15.78
    map_center_lon = df_filtered[lon_col].mean() if not df_filtered[lon_col].empty else -47.93

    fig.update_layout(
        title=title,
        mapbox_style="open-street-map", # MODIFICADO AQUI
        mapbox_center_lat=map_center_lat,
        mapbox_center_lon=map_center_lon,
        mapbox_zoom=3.5,
        margin={"r":0,"t":50,"l":0,"b":0},
        height=700,
        legend_title_text='<b>Tipo de Evento (Cor)</b><br><i>Severidade do Risco (Tamanho do Ponto)</i>'
    )
    fig.show()


    
# --- 5. Workflow Principal ---
def main():
    """Executa o workflow completo de modelagem."""
    # MODIFICADO: load_and_prepare_data agora retorna 3 DataFrames
    X_model_df, y_series, X_map_info_df = load_and_prepare_data()
    
    if X_model_df is None or X_model_df.empty: return
    if y_series is None or y_series.empty: return

    # Identificar tipos de feature APENAS para X_model_df
    numerical_features, categorical_features = identify_feature_types(X_model_df)
    
    if not numerical_features and not categorical_features:
        print("Erro: Nenhuma feature para o modelo identificada.")
        return
    
    preprocessor = get_preprocessor(numerical_features, categorical_features)

    # MODIFICADO: Fazer o split dos 3 DataFrames/Series
    X_train_model, X_test_model, \
    y_train, y_test, \
    X_train_map_info, X_test_map_info = train_test_split(
        X_model_df,
        y_series,
        X_map_info_df, # Inclui lat, lon, tipo_evento e outras infos para o mapa
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_series # Estratificar pela variável alvo
    )
    print(f"\nDados divididos: X_train_model ({X_train_model.shape}), X_test_model ({X_test_model.shape}), X_test_map_info ({X_test_map_info.shape})")
    
    # ... (cálculo de scale_pos_weight_val como antes, usando y_train) ...
    neg_count, pos_count = np.bincount(y_train) if len(y_train) > 0 and len(np.bincount(y_train)) > 1 else (1,1)
    scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1

    models_to_run = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, class_weight='balanced', solver='liblinear'),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample'),
        "XGBoost": xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight_val),
        "LightGBM": lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, class_weight='balanced'),
        "CatBoost": cb.CatBoostClassifier(random_state=RANDOM_STATE, verbose=0, scale_pos_weight=scale_pos_weight_val)
    }

    print("\n--- Treinamento e Avaliação dos Modelos ---")
    trained_pipelines = {}
    model_to_map = "XGBoost"

    for name, model_instance in models_to_run.items(): # MODIFICADO: model_instance
        print(f"\nTreinando e avaliando {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model_instance)]) # MODIFICADO: model_instance
        
        start_time = time.time()
        try: pipeline.fit(X_train_model, y_train) # MODIFICADO: usa X_train_model
        except Exception as e: print(f"Erro treino {name}: {e}"); continue
        duration = time.time() - start_time
        print(f"Treino {name} em {duration:.2f}s.")
        
        y_pred_test, y_pred_proba_test = None, None
        try:
            y_pred_test = pipeline.predict(X_test_model) # MODIFICADO: usa X_test_model
            y_pred_proba_test = pipeline.predict_proba(X_test_model)[:, 1]
        except Exception as e: print(f"Erro predição {name}: {e}"); continue

        evaluate_model(y_test, y_pred_test, y_pred_proba_test, model_name=name)
        trained_pipelines[name] = pipeline

        if ENABLE_STATIC_PLOTS: # MODIFICADO: Chamada condicional
            if name in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]:
                try: display_feature_importance(pipeline, X_train_model, model_name=name) # MODIFICADO: X_train_model
                except Exception as e: print(f"Erro feat importance {name}: {e}")

        if name == model_to_map and y_pred_proba_test is not None:
            print(f"\n--- Gerando Mapa de Classificação para {name} ---")
            # MODIFICADO: df_for_map agora usa X_test_map_info e adiciona a probabilidade predita
            df_for_map = X_test_map_info.copy() # Contém lat, lon, tipo_evento original, e outras infos
            df_for_map['probability'] = y_pred_proba_test # Adiciona probabilidade predita pelo modelo

            display_classification_map(
                df_for_map,
                lat_col='lat', 
                lon_col='lon',
                proba_col='probability',
                event_type_col='tipo_evento', # Coluna do tipo de evento original
                title=f"Mapa de Risco ({name}) - Cor: Tipo Evento, Tamanho: Severidade"
            )

    if ENABLE_STATIC_PLOTS and SHAP_AVAILABLE and model_to_map in trained_pipelines: # MODIFICADO
        print(f"\n--- Interpretabilidade com SHAP para {model_to_map} ---")
        chosen_pipeline = trained_pipelines[model_to_map]
        explain_with_shap(chosen_pipeline, X_test_model, model_name=model_to_map) # MODIFICADO: X_test_model
    
    print("\n--- Fim da Execução ---")

if __name__ == '__main__':
    main()