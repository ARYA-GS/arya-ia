import pandas as pd
import numpy as np
import time

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
def display_classification_map(df_map_with_preds_and_info, lat_col='lat', lon_col='lon',
                               proba_col='probability', event_type_col='tipo_evento',
                               title="Mapa de Classificação de Risco por Tipo de Evento"):
    """
    Exibe um mapa interativo com as classificações de risco.
    df_map_with_preds_and_info: DataFrame contendo 'lat', 'lon', 'tipo_evento' (original),
                                 e 'probability' (predita pelo modelo), e outras features para hover.
    """
    print(f"Gerando mapa. Amostra dos dados para o mapa (antes de filtrar 'nenhum'):\n{df_map_with_preds_and_info.head()}")

    # --- NOVO: Filtrar eventos do tipo "nenhum" ---
    df_filtered_for_map = df_map_with_preds_and_info[df_map_with_preds_and_info[event_type_col] != 'nenhum'].copy()
    # Usar .copy() para evitar SettingWithCopyWarning em operações subsequentes

    if df_filtered_for_map.empty:
        print("Nenhum evento para exibir no mapa após filtrar 'nenhum'.")
        return
    
    print(f"Amostra dos dados para o mapa (APÓS filtrar 'nenhum'):\n{df_filtered_for_map.head()}")


    # 1. Definir Nivel_Risco (severidade) baseado na probabilidade predita
    bins_risco = [-0.01, 0.3, 0.7, 1.01]
    labels_risco = ['Baixo Risco', 'Médio Risco', 'Alto Risco']
    df_filtered_for_map['Nivel_Risco_Predito'] = pd.cut(
        df_filtered_for_map[proba_col], bins=bins_risco, labels=labels_risco, right=True, include_lowest=True
    )

    # 2. Definir mapa de cores para event_type_col (tipo de evento original)
    default_event_colors = {
        'inundação': 'rgba(0, 100, 255, 0.8)',    # Azul mais vibrante
        'deslizamento': 'rgba(165, 42, 42, 0.8)', # Marrom (Brown)
        'seca': 'rgba(255, 193, 7, 0.8)',        # Ambar/Laranja para seca
        'incêndio': 'rgba(220, 53, 69, 0.8)',     # Vermelho mais forte
        'vendaval': 'rgba(128, 0, 128, 0.7)'     # Roxo para vendaval
    }
    unique_event_types_on_map = df_filtered_for_map[event_type_col].unique()
    event_color_map = {
        etype: default_event_colors.get(etype, 'rgba(108, 117, 125, 0.7)') # Cinza secundário para não mapeados
        for etype in unique_event_types_on_map
    }

    # 3. Definir tamanho do marcador (efeito de raio) baseado no Nivel_Risco_Predito
    # MODIFICADO: Aumentando AINDA MAIS os tamanhos para melhor visualização do "raio"
    size_map = {'Baixo Risco': 12, 'Médio Risco': 22, 'Alto Risco': 30} # Valores anteriores: 10, 18, 25
    
    mapped_sizes = df_filtered_for_map['Nivel_Risco_Predito'].map(size_map)
    df_filtered_for_map['marker_size'] = mapped_sizes.astype(float).fillna(10.0).astype(int) # Usar 10 como fallback

    # 4. Configurar hover_data
    hover_data_list = [
        event_type_col,
        'Nivel_Risco_Predito',
        proba_col,
        'chuva_mm',
        'temperatura_media',
        'uso_solo'
    ]
    hover_data_list_filtered = [col for col in hover_data_list if col in df_filtered_for_map.columns]
    
    # Não precisamos mais do df_display separado, pois df_filtered_for_map já é uma cópia
    if proba_col in df_filtered_for_map.columns:
        df_filtered_for_map[proba_col] = df_filtered_for_map[proba_col].round(3)

    fig = px.scatter_mapbox(
        df_filtered_for_map, # MODIFICADO: Usar o DataFrame filtrado
        lat=lat_col,
        lon=lon_col,
        color=event_type_col,
        color_discrete_map=event_color_map,
        size='marker_size', 
        zoom=3.5,
        hover_name=event_type_col,
        hover_data=hover_data_list_filtered,
        title=title
    )

    map_center_lat = df_filtered_for_map[lat_col].mean() if not df_filtered_for_map[lat_col].empty else -15.78
    map_center_lon = df_filtered_for_map[lon_col].mean() if not df_filtered_for_map[lon_col].empty else -47.93

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center_lat=map_center_lat,
        mapbox_center_lon=map_center_lon,
        margin={"r":0,"t":50,"l":0,"b":0},
        height=700,
        legend_title_text='Tipo de Evento (Cor)<br>Severidade (Tamanho)'
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