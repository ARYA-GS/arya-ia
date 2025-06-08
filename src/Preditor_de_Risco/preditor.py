import pandas as pd
import numpy as np
import time
import geopandas as gpd
import plotly.graph_objects as go
from datetime import datetime
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    make_scorer,
    f1_score, recall_score, precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Biblioteca SHAP não instalada. A interpretabilidade com SHAP será pulada.")


RANDOM_STATE = 42
TARGET_COLUMN = 'nivel_de_risco'
ENABLE_STATIC_PLOTS = False
DO_HYPERPARAMETER_TUNING = False 
DO_OVERSAMPLING = True

RUN_ALL_MODELS = False 


def load_and_prepare_data(filepath='C:/Users/zenet/OneDrive/Desktop/ARYA_IA_GS/arya-ia/Data/CSV/desastres_naturais_20250608_141114.csv'):
    """Carrega os dados e faz uma preparação inicial e engenharia de features de data."""
    try:
        df_full = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{filepath}' não encontrado. Por favor, verifique o caminho.")
        raise
    
    print("Dados carregados com sucesso.")
    print(f"Shape inicial: {df_full.shape}")

    if TARGET_COLUMN not in df_full.columns:
        raise KeyError(f"Erro: A coluna alvo '{TARGET_COLUMN}' não foi encontrada no arquivo CSV. "
                       f"Verifique a grafia exata (incluindo maiúsculas/minúsculas e espaços).")


    if 'data' in df_full.columns:
        df_full['data'] = pd.to_datetime(df_full['data'], errors='coerce')
        df_full['ano'] = df_full['data'].dt.year
        df_full['dia_do_ano'] = df_full['data'].dt.dayofyear
        df_full['dia_da_semana'] = df_full['data'].dt.dayofweek
        df_full['trimestre'] = df_full['data'].dt.quarter
        df_full.drop(columns=['data'], inplace=True)


    cols_to_drop_for_model = ['id_zona', TARGET_COLUMN] 

    cols_present_to_drop_for_model = [col for col in cols_to_drop_for_model if col in df_full.columns]
    
    X_model_df = df_full.drop(columns=cols_present_to_drop_for_model, errors='ignore')
    y_series = df_full[TARGET_COLUMN]
    

    map_info_cols = ['latitude', 'longitude', 'tipo_evento', 'precipitacao_mm',
                     'temperatura_c', 'uso_do_solo', 'ocorrenca',
                     'ano', 'dia_do_ano', 'dia_da_semana', 'trimestre', 'nivel_de_risco',
                     'umidade_percentual', 'densidade_populacional', 'altitude_metros',
                     'declividade_graus', 'distancia_agua_km', 'frequencia_sismos',
                     'tipo_de_solo', 'nivel_acessibilidade', 'estacao_do_ano', 'regiao', 'mes']
    
    map_info_cols_present = [col for col in map_info_cols if col in df_full.columns]
    X_map_info_df = df_full[map_info_cols_present].copy()


    for col in X_map_info_df.columns:
        if pd.api.types.is_numeric_dtype(X_map_info_df[col]):
            X_map_info_df[col].fillna(X_map_info_df[col].mean(), inplace=True)
        elif X_map_info_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_map_info_df[col]):
            X_map_info_df[col] = X_map_info_df[col].astype(str).str.strip() 
            X_map_info_df[col].fillna('N/A', inplace=True) 
            X_map_info_df[col] = X_map_info_df[col].replace('nan', 'N/A') 


    print(f"Shape de X_model_df (features para o modelo): {X_model_df.shape}")
    print(f"Shape de X_map_info_df (infos para o mapa): {X_map_info_df.shape}")
    return X_model_df, y_series, X_map_info_df


def identify_feature_types(df_for_model_features):
    numerical_features = [
        'latitude', 'longitude', 'precipitacao_mm', 'temperatura_c',
        'umidade_percentual', 'densidade_populacional', 'altitude_metros',
        'declividade_graus', 'distancia_agua_km', 'frequencia_sismos',
        'ano', 'dia_do_ano', 'dia_da_semana', 'trimestre', 'ocorrenca'
    ]
    categorical_features = [
        'tipo_de_solo', 'uso_do_solo', 'nivel_acessibilidade',
        'tipo_evento', 'estacao_do_ano', 'regiao', 'mes'
    ]
    
    numerical_features = [col for col in numerical_features if col in df_for_model_features.columns]
    categorical_features = [col for col in categorical_features if col in df_for_model_features.columns]

    for col in numerical_features:
        if col in df_for_model_features.columns:
            df_for_model_features[col] = pd.to_numeric(df_for_model_features[col], errors='coerce')
            df_for_model_features[col].fillna(df_for_model_features[col].mean(), inplace=True)

    for col in categorical_features:
        if col in df_for_model_features.columns:
            df_for_model_features[col] = df_for_model_features[col].astype(str).str.strip().fillna('MISSING_CATEGORY')

    print(f"Features numéricas (modelo): {numerical_features}")
    print(f"Features categóricas (modelo): {categorical_features}")
    return numerical_features, categorical_features


def get_preprocessor(numerical_features, categorical_features):
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor


def evaluate_model(y_true, y_pred, y_pred_proba, model_name="Model"):
    print(f"\n--- Resultados para {model_name} ---")
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.4f}")
    
    unique_classes_true = np.unique(y_true)
    num_classes = len(unique_classes_true)

    if num_classes > 2:
        print("Métricas ROC AUC e PR AUC não calculadas para classificação multiclasse > 2. Focando em Acurácia e Classification Report.")
    elif num_classes == 2 and y_pred_proba is not None and y_pred_proba.ndim == 1:
        try:
            print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
        except ValueError:
            print("ROC AUC não pode ser calculado (valores em y_pred_proba não válidos para ROC AUC).")
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        print(f"PR AUC: {pr_auc:.4f}")
    else:
        print("ROC AUC e PR AUC não podem ser calculados para o formato de y_pred_proba ou número de classes.")
    
    print("\nMatriz de Confusão:\n", confusion_matrix(y_true, y_pred))
    print("\nRelatório de Classificação:")
    sorted_classes = sorted(unique_classes_true)
    target_names_report = [str(c) for c in sorted_classes]
        
    print(classification_report(y_true, y_pred, target_names=target_names_report, zero_division=0))



def get_feature_names_from_preprocessor(preprocessor, X_cols_original_names):
    output_features = []
    for name, transformer, original_cols_applied in preprocessor.transformers:
        if name == 'num':
            output_features.extend(original_cols_applied)
        elif name == 'cat':
            if hasattr(transformer, 'named_steps') and 'onehot' in transformer.named_steps:
                ohe = transformer.named_steps['onehot']
                output_features.extend(ohe.get_feature_names_out(original_cols_applied))
            else:
                output_features.extend([f"cat_{col}_{i}" for col in original_cols_applied for i in range(len(pd.Series(original_cols_applied).unique()))])
        elif name == 'remainder' and transformer != 'drop':
            if hasattr(transformer, 'get_feature_names_out'):
                output_features.extend(transformer.get_feature_names_out(original_cols_applied))
            else:
                output_features.extend(original_cols_applied)

    if not output_features:
        print("Aviso: Não foi possível inferir nomes de features processadas. Usando nomes originais.")
        return X_cols_original_names.columns.tolist()
    return output_features


def display_feature_importance(pipeline, X_original_features_df, model_name="Model"):
    if not ENABLE_STATIC_PLOTS: return
    
    classifier = pipeline.named_steps['classifier']

    if hasattr(classifier, 'feature_importances_'):
        feature_names_processed = get_feature_names_from_preprocessor(pipeline.named_steps['preprocessor'], X_original_features_df)
        
        importances = classifier.feature_importances_
        
        if len(feature_names_processed) != len(importances):
            print(f"Aviso (display_feature_importance): Mismatch de tamanho: {len(feature_names_processed)} features vs {len(importances)} importances. Plot pode ser impreciso.")
            if len(feature_names_processed) > len(importances):
                feature_names_processed = feature_names_processed[:len(importances)]
            else:
                importances = importances[:len(feature_names_processed)]


        if len(feature_names_processed) == len(importances) and feature_names_processed:
            feature_importance_df = pd.DataFrame({'feature': feature_names_processed, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title(f'Top 20 Importância de Features ({model_name})')
            plt.tight_layout()
            plt.show(block=False)
        else:
            print("Não foi possível plotar a importância das features devido a inconsistência de tamanho ou lista vazia.")
    else:
        print(f"Modelo {model_name} não possui attribute 'feature_importances_'.")


def explain_with_shap(pipeline, X_test_model_features, model_name="Model"):
    if not ENABLE_STATIC_PLOTS or not SHAP_AVAILABLE: return

    try:
        X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test_model_features)
        feature_names_processed = get_feature_names_from_preprocessor(pipeline.named_steps['preprocessor'], X_test_model_features)

        classifier = pipeline.named_steps['classifier']

        if isinstance(classifier, (xgb.XGBClassifier, lgb.LGBMClassifier, cb.CatBoostClassifier, RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(classifier)
        else:
            if X_test_processed.shape[0] > 100:
                sample_indices = np.random.choice(X_test_processed.shape[0], 100, replace=False)
                X_test_sample = X_test_processed[sample_indices]
            else:
                X_test_sample = X_test_processed

            if not hasattr(classifier, "predict_proba"):
                print(f"SHAP KernelExplainer requer o método predict_proba para o modelo {model_name}.")
                return
            
            explainer = shap.KernelExplainer(classifier.predict_proba, X_test_sample)

        shap_values = explainer.shap_values(X_test_processed)

        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[0] 
            print(f"Plotando SHAP para a primeira classe (índice 0) dos valores SHAP.")
        else:
            shap_values_to_plot = shap_values


        shap.summary_plot(shap_values_to_plot, X_test_processed, feature_names=feature_names_processed, show=False)
        plt.title(f"SHAP Summary Plot ({model_name})")
        plt.show(block=False)

        shap.summary_plot(shap_values_to_plot, X_test_processed, feature_names=feature_names_processed, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance Bar Plot ({model_name})")
        plt.show(block=False)

    except Exception as e:
        print(f"Erro ao gerar plots SHAP para {model_name}: {e}")

    plt.show(block=False)


def display_classification_map(df_map_with_preds_and_info, lat_col='latitude', lon_col='longitude',
                               proba_col='predicted_level',
                               event_type_col='tipo_evento',
                               title="Mapa de Classificação de Nível de Risco"):
    """
    Exibe um mapa interativo com pontos sólidos e halos transparentes,
    agora focado na classificação do Nível de Risco.
    Retorna a figura Plotly.
    """
    print(f"Gerando mapa com Plotly Graph Objects. Amostra dos dados:\n{df_map_with_preds_and_info.head(3)}")

    df_filtered = df_map_with_preds_and_info.dropna(subset=[lat_col, lon_col, event_type_col, proba_col]).copy()
    df_filtered = df_filtered[df_filtered[event_type_col] != 'nenhum'].copy()
    
    if df_filtered.empty:
        print("Nenhum evento relevante para exibir no mapa após filtrar ou devido a NaNs.")
        return None

    df_filtered[lat_col] = pd.to_numeric(df_filtered[lat_col], errors='coerce')
    df_filtered[lon_col] = pd.to_numeric(df_filtered[lon_col], errors='coerce')
    df_filtered.dropna(subset=[lat_col, lon_col], inplace=True)

    if df_filtered.empty:
        print("Nenhum dado válido de latitude/longitude para centralizar o mapa após conversão.")
        return None


    cor_por_risco_predito = {
        'MUITO BAIXO': '#a6d96a', 
        'BAIXO': '#66bd63',      
        'MODERADO': '#ffffbf',    
        'MÉDIO': '#fdae61',       
        'ALTO': '#f46d43',        
        'CRITICO': '#d73027',    
        'N/A': 'grey'             
    }


    tamanho_por_risco_predito = {
        'MUITO BAIXO': {'solid': 6, 'halo': 25},
        'BAIXO': {'solid': 8, 'halo': 30},
        'MODERADO': {'solid': 10, 'halo': 38},
        'MÉDIO': {'solid': 12, 'halo': 45},
        'ALTO': {'solid': 16, 'halo': 60},
        'CRITICO': {'solid': 20, 'halo': 75},
        'N/A': {'solid': 7, 'halo': 25} 
    }
    
    df_filtered['color_predito'] = df_filtered[proba_col].map(lambda x: cor_por_risco_predito.get(str(x).strip(), 'grey'))
    df_filtered['solid_marker_size'] = df_filtered[proba_col].apply(lambda x: tamanho_por_risco_predito.get(str(x).strip(), {'solid': 7})['solid']).astype(int)
    df_filtered['halo_marker_size'] = df_filtered[proba_col].apply(lambda x: tamanho_por_risco_predito.get(str(x).strip(), {'halo': 20})['halo']).astype(int)

    hover_texts = []
    for _, row in df_filtered.iterrows():
        def format_value(val):
            if pd.isna(val) or val is None:
                return 'N/A'
            if isinstance(val, (float, np.float32, np.float64)):
                if val == int(val):
                    return str(int(val))
                return f"{val:.1f}"
            return str(val)


        text = (f"<b>Nível de Risco Predito:</b> {format_value(row[proba_col])}<br>"
                f"<b>Tipo Evento:</b> {format_value(row.get('tipo_evento', 'N/A')).capitalize()}<br>"
                f"Lat: {format_value(row[lat_col])}, Lon: {format_value(row[lon_col])}<br>")
        

        text += f"Ocorrência (original): {format_value(row.get('ocorrenca', 'N/A'))}<br>"
        text += f"Precipitação (mm): {format_value(row.get('precipitacao_mm', 'N/A'))}<br>"
        text += f"Temperatura (°C): {format_value(row.get('temperatura_c', 'N/A'))}<br>"
        text += f"Umidade (%): {format_value(row.get('umidade_percentual', 'N/A'))}<br>"
        text += f"Densidade Pop.: {format_value(row.get('densidade_populacional', 'N/A'))}<br>"
        text += f"Altitude (m): {format_value(row.get('altitude_metros', 'N/A'))}<br>"
        text += f"Declividade (°): {format_value(row.get('declividade_graus', 'N/A'))}<br>"
        text += f"Dist. Água (km): {format_value(row.get('distancia_agua_km', 'N/A'))}<br>"
        text += f"Tipo Solo: {format_value(row.get('tipo_de_solo', 'N/A'))}<br>"
        text += f"Uso do Solo: {format_value(row.get('uso_do_solo', 'N/A'))}<br>"
        text += f"Nível Acessibilidade: {format_value(row.get('nivel_acessibilidade', 'N/A'))}<br>" 
        text += f"Freq. Sismos: {format_value(row.get('frequencia_sismos', 'N/A'))}<br>"
        text += f"Mês: {format_value(row.get('mes', 'N/A'))}<br>"
        text += f"Estação: {format_value(row.get('estacao_do_ano', 'N/A'))}<br>"
        text += f"Região: {format_value(row.get('regiao', 'N/A'))}<br>"
        text += f"Ano: {format_value(row.get('ano', 'N/A'))}<br>"
        text += f"Dia do Ano: {format_value(row.get('dia_do_ano', 'N/A'))}<br>"
        text += f"Dia da Semana: {format_value(row.get('dia_da_semana', 'N/A'))}<br>"
        text += f"Trimestre: {format_value(row.get('trimestre', 'N/A'))}"

        hover_texts.append(text)
    df_filtered['hover_text'] = hover_texts

    fig = go.Figure()


    ordered_risk_levels_for_legend = [
        'CRITICO', 'ALTO', 'MÉDIO', 'MODERADO', 'BAIXO', 'MUITO BAIXO', 'N/A' 
    ]
    

    existing_unique_levels = [level for level in ordered_risk_levels_for_legend if level in df_filtered[proba_col].unique()]
    

    for level in df_filtered[proba_col].unique():
        if level not in existing_unique_levels:
            existing_unique_levels.append(level)

    for risk_level_val in existing_unique_levels:
        df_subset = df_filtered[df_filtered[proba_col] == risk_level_val]
        if df_subset.empty:
            continue
        
        current_color = cor_por_risco_predito.get(str(risk_level_val).strip(), 'grey')
        
        fig.add_trace(go.Scattermapbox(
            lat=df_subset[lat_col],
            lon=df_subset[lon_col],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=df_subset['halo_marker_size'],
                color=current_color,
                opacity=0.12
            ),
            hoverinfo='skip',
            name=f'Halo: {risk_level_val}',
            showlegend=False
        ))

        fig.add_trace(go.Scattermapbox(
            lat=df_subset[lat_col],
            lon=df_subset[lon_col],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=df_subset['solid_marker_size'],
                color=current_color,
                opacity=0.9
            ),
            hovertext=df_subset['hover_text'],
            hoverinfo='text',
            name=f'Nível de Risco: {risk_level_val}',
            showlegend=True
        ))
    
    map_center_lat = df_filtered[lat_col].mean()
    map_center_lon = df_filtered[lon_col].mean()

    fig.update_layout(
        title=title,
        mapbox_style="open-street-map", 
        mapbox_center_lat=map_center_lat,
        mapbox_center_lon=map_center_lon,
        mapbox_zoom=4.5, 
        margin={"r": 0, "t": 0, "l": 0, "b": 0}, 
        legend=dict(
            title_text='<b>Classificação de Risco (Cor e Tamanho)</b>',
            x=0.01, y=0.99,
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='Black',
            borderwidth=1
        )
    )
    return fig


def main_preditor():
    filepath = "C:/Users/zenet/OneDrive/Desktop/GS_IA_FINAL/arya-ia/Data/CSV/desastres_naturais_20250608_141114.csv"

    X_model_df, y_series, X_map_info_df = load_and_prepare_data(filepath)
    
    if X_model_df is None or X_model_df.empty or y_series is None or y_series.empty:
        print("Dados insuficientes para prosseguir com o treinamento do modelo.")
        return None

    numerical_features, categorical_features = identify_feature_types(X_model_df)
    
    print(f"Features numéricas finais para o modelo: {numerical_features}")
    print(f"Features categóricas finais para o modelo: {categorical_features}")

    if not numerical_features and not categorical_features:
        print("Erro: Nenhuma feature válida para o modelo identificada.")
        return None
    
    preprocessor = get_preprocessor(numerical_features, categorical_features)

    label_encoder_target = LabelEncoder()
    y_encoded = label_encoder_target.fit_transform(y_series)
    
    class_counts = pd.Series(y_encoded).value_counts()
    min_class_count = class_counts.min()
    num_unique_classes = len(class_counts)


    if min_class_count < 2:
        print(f"/nAviso: A classe menos populosa em '{TARGET_COLUMN}' (após encoding) tem apenas {min_class_count} membro(s).")
        print("A estratificação no train_test_split requer no mínimo 2 membros por classe. Desabilitando estratificação.")
        X_train_model, X_test_model, \
        y_train_encoded, y_test_encoded, \
        X_train_map_info, X_test_map_info = train_test_split(
            X_model_df,
            y_encoded,
            X_map_info_df,
            test_size=0.25,
            random_state=RANDOM_STATE,

        )
    else:
        X_train_model, X_test_model, \
        y_train_encoded, y_test_encoded, \
        X_train_map_info, X_test_map_info = train_test_split(
            X_model_df,
            y_encoded,
            X_map_info_df,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_encoded
        )

    print(f"\nDados divididos: X_train_model ({X_train_model.shape}), X_test_model ({X_test_model.shape}), X_test_map_info ({X_test_map_info.shape})")
    
    y_train = y_train_encoded

    if DO_OVERSAMPLING and num_unique_classes > 1 and min_class_count >= 2:
        print("\nSMOTE será aplicado DENTRO do pipeline para balanceamento de classes no conjunto de treino.")
        smote_step = ('sampler', SMOTE(random_state=RANDOM_STATE, k_neighbors=min(min_class_count - 1, 5)))
    else:
        print(f"\nSMOTE desativado: Não aplicável para {num_unique_classes} classe(s) ou classe minoritária ({min_class_count} amostra(s)) é muito pequena.")
        smote_step = None

    models_to_run = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=10000, class_weight='balanced', solver='liblinear'),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample'),
        "XGBoost": xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss' if num_unique_classes > 2 else 'logloss'),
        "LightGBM": lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, class_weight='balanced'),
        "CatBoost": cb.CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
    }

    print("\n--- Treinamento e Avaliação dos Modelos ---")
    trained_pipelines = {}
    model_to_map = "XGBoost"

    models_to_process = models_to_run.keys() if RUN_ALL_MODELS else [model_to_map]

    for name in models_to_process:
        model_instance = models_to_run[name]

        print(f"\nTreinando e avaliando {name}...")
        
        steps = [('preprocessor', preprocessor)]
        if smote_step:
            steps.append(smote_step)
        steps.append(('classifier', model_instance))
        
        pipeline = ImbPipeline(steps=steps)

        if DO_HYPERPARAMETER_TUNING:
            print(f"Iniciando otimização de hiperparâmetros para {name}...")
            param_grid = {}
            scorer = make_scorer(f1_score, average='weighted')
            
            if name == "Random Forest":
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [5, 10, None]
                }
            elif name == "XGBoost":
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2]
                }
            
            if param_grid:
                grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring=scorer, n_jobs=-1, verbose=0)
                grid_search.fit(X_train_model, y_train_encoded)
                pipeline = grid_search.best_estimator_
                print(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
            else:
                print(f"Nenhum grid de parâmetros definido para {name}. Treinando com parâmetros padrão.")

        start_time = time.time()
        try:
            pipeline.fit(X_train_model, y_train_encoded)
        except Exception as e:
            print(f"Erro no treino do modelo {name}: {e}")
            continue
        duration = time.time() - start_time
        print(f"Treino do modelo {name} concluído em {duration:.2f} segundos.")
        
        y_pred_test_encoded = None
        y_pred_proba_test = None
        try:
            y_pred_test_encoded = pipeline.predict(X_test_model)
            if hasattr(pipeline, 'predict_proba') and num_unique_classes > 1:
                y_pred_proba_test = pipeline.predict_proba(X_test_model)
            else:
                y_pred_proba_test = None

        except Exception as e:
            print(f"Erro na predição do modelo {name}: {e}")
            continue

        y_test_original = label_encoder_target.inverse_transform(y_test_encoded)
        y_pred_test_original = label_encoder_target.inverse_transform(y_pred_test_encoded)

        evaluate_model(y_test_original, y_pred_test_original, y_pred_proba_test, model_name=name)
        trained_pipelines[name] = pipeline
        
    if ENABLE_STATIC_PLOTS and SHAP_AVAILABLE and model_to_map in trained_pipelines:
        print(f"\n--- Interpretabilidade com SHAP para {model_to_map} ---")
        chosen_pipeline = trained_pipelines[model_to_map]
        explain_with_shap(chosen_pipeline, X_test_model, model_name=model_to_map)
    
    if model_to_map in trained_pipelines:
        print(f"\n--- Gerando Mapa de Classificação para {model_to_map} ---")
        chosen_pipeline = trained_pipelines[model_to_map]
        
        y_pred_map_encoded = chosen_pipeline.predict(X_test_model)
        y_pred_map_original = label_encoder_target.inverse_transform(y_pred_map_encoded)

        df_for_map = X_test_map_info.copy()
        df_for_map['predicted_level'] = y_pred_map_original

        return display_classification_map(
            df_for_map,
            lat_col='latitude',
            lon_col='longitude',
            proba_col='predicted_level',
            event_type_col='tipo_evento',
            title=f"Mapa de Classificação de Nível de Risco ({model_to_map})"
        )
    else:
        print(f"O modelo '{model_to_map}' não foi treinado com sucesso.")
        return None