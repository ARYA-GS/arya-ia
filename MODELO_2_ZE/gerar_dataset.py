import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configurações
n_amostras = 2000
# zonas = [f'ZONA_{i:03d}' for i in range(1, n_amostras + 1)] # id_zona será gerado dentro da função
tipos_solo_geral = ['argiloso', 'arenoso', 'rochoso', 'organico'] # Adicionado 'organico'
usos_solo_geral = ['urbano', 'rural', 'floresta', 'pastagem', 'area_umida'] # Adicionado 'pastagem' e 'area_umida'
tipos_evento_geral = ['inundação', 'deslizamento', 'seca', 'incêndio', 'vendaval'] # Adicionado 'vendaval'

# Definição de Macro-Regiões Brasileiras com suas características e limites aproximados
# (Limites são aproximados e podem se sobrepor ou ter lacunas para simplificação)
REGIOES_BRASIL = {
    "Norte_Amazonia": {
        "lat_range": (-10.0, 5.0), # Latitude: de 10°S a 5°N
        "lon_range": (-74.0, -48.0), # Longitude: de 74°W a 48°W
        "chuva_range": (150, 450), # mm
        "temp_range": (22, 38),   # °C
        "umidade_range": (70, 100), # %
        "altitude_range": (0, 500),  # m
        "declive_range": (0, 20),   # graus
        "tipos_solo": ['argiloso', 'organico', 'arenoso'],
        "usos_solo": ['floresta', 'area_umida', 'rural', 'urbano'], # Urbano menos frequente
        "eventos_comuns": ['inundação', 'vendaval', 'incêndio'], # Incêndio por desmatamento/seca esporádica
        "prob_peso": 0.25 # Probabilidade de escolher esta região
    },
    "Nordeste_Sertao": {
        "lat_range": (-18.0, -2.0),
        "lon_range": (-46.0, -37.0),
        "chuva_range": (5, 100), # Baixa chuva característica
        "temp_range": (20, 42),
        "umidade_range": (20, 70),
        "altitude_range": (100, 1000),
        "declive_range": (0, 30),
        "tipos_solo": ['arenoso', 'rochoso', 'argiloso_seco'], # argiloso_seco como variação
        "usos_solo": ['rural', 'pastagem'], # Vegetação de caatinga, pouca floresta densa
        "eventos_comuns": ['seca', 'incêndio'],
        "prob_peso": 0.20
    },
    "Sudeste_Montanhoso_Litoraneo": {
        "lat_range": (-25.0, -18.0),
        "lon_range": (-50.0, -39.0),
        "chuva_range": (80, 350), # Chuvas orográficas
        "temp_range": (15, 35),
        "umidade_range": (50, 95),
        "altitude_range": (0, 2800), # Inclui litoral e serras
        "declive_range": (5, 60),    # Áreas de encosta íngremes
        "tipos_solo": ['argiloso', 'rochoso', 'organico'],
        "usos_solo": ['urbano', 'floresta', 'rural', 'pastagem'], # Mata Atlântica, áreas urbanas densas
        "eventos_comuns": ['deslizamento', 'inundação', 'vendaval'],
        "prob_peso": 0.25
    },
    "Sul_Campos_Planalto": {
        "lat_range": (-34.0, -22.0),
        "lon_range": (-58.0, -48.0),
        "chuva_range": (70, 250), # Bem distribuída ou concentrada dependendo da sub-região
        "temp_range": (5, 32),  # Maior amplitude térmica
        "umidade_range": (40, 90),
        "altitude_range": (0, 1800),
        "declive_range": (0, 40),
        "tipos_solo": ['argiloso', 'arenoso_argiloso', 'organico'], # Solos férteis
        "usos_solo": ['rural', 'pastagem', 'floresta', 'urbano'], # Campos sulinos, Araucárias
        "eventos_comuns": ['inundação', 'vendaval', 'seca'], # Secas no verão podem ocorrer
        "prob_peso": 0.15
    },
    "CentroOeste_Cerrado_Pantanal": {
        "lat_range": (-24.0, -5.0),
        "lon_range": (-61.0, -45.0),
        "chuva_range": (60, 200), # Sazonal, verões chuvosos, invernos secos
        "temp_range": (18, 38),
        "umidade_range": (30, 85), # Varia muito com a estação
        "altitude_range": (50, 1200), # Inclui planícies do Pantanal e planaltos do Cerrado
        "declive_range": (0, 25),
        "tipos_solo": ['argiloso', 'arenoso', 'latossolo'], # Latossolo é comum no Cerrado
        "usos_solo": ['rural', 'pastagem', 'floresta', 'area_umida'], # Cerrado, Pantanal
        "eventos_comuns": ['incêndio', 'inundação', 'seca'], # Incêndios no inverno, inundações no Pantanal
        "prob_peso": 0.15
    }
}

# Adicionar 'argiloso_seco' e 'arenoso_argiloso', 'latossolo' à lista geral para colunas do DataFrame
tipos_solo_geral = list(set(tipos_solo_geral + ['argiloso_seco', 'arenoso_argiloso', 'latossolo']))


def gerar_linha_por_regiao(id_zona):
    # 1. Escolher uma região com base nos pesos
    nomes_regioes = list(REGIOES_BRASIL.keys())
    pesos = [REGIOES_BRASIL[r]["prob_peso"] for r in nomes_regioes]
    regiao_escolhida_nome = random.choices(nomes_regioes, weights=pesos, k=1)[0]
    regiao = REGIOES_BRASIL[regiao_escolhida_nome]

    # 2. Gerar coordenadas dentro da região
    lat = round(np.random.uniform(regiao["lat_range"][0], regiao["lat_range"][1]), 5)
    lon = round(np.random.uniform(regiao["lon_range"][0], regiao["lon_range"][1]), 5)
    
    # 3. Gerar data e estação
    data = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365*2)) # Aumentado para 2 anos
    mes = data.month
    # Estação do ano (simplificada para Hemisfério Sul)
    if mes in [12, 1, 2]: estacao = 'verao' # Verão (mais chuvoso em muitas partes)
    elif mes in [3, 4, 5]: estacao = 'outono'
    elif mes in [6, 7, 8]: estacao = 'inverno' # Inverno (mais seco em muitas partes)
    else: estacao = 'primavera'

    # 4. Gerar features com base na região e estação
    chuva = round(np.random.uniform(regiao["chuva_range"][0], regiao["chuva_range"][1]), 1)
    if regiao_escolhida_nome == "CentroOeste_Cerrado_Pantanal" or regiao_escolhida_nome == "Nordeste_Sertao":
        if estacao == 'inverno': # Invernos secos no Cerrado e Sertão
            chuva = round(np.random.uniform(0, chuva * 0.3), 1) # Reduz muito a chuva
        elif estacao == 'verao': # Verões chuvosos
             chuva = round(np.random.uniform(chuva * 0.7, chuva * 1.2), 1)


    temperatura = round(np.random.uniform(regiao["temp_range"][0], regiao["temp_range"][1]), 1)
    umidade = round(np.random.uniform(regiao["umidade_range"][0], regiao["umidade_range"][1]), 1)
    if estacao == 'inverno' and regiao_escolhida_nome != "Norte_Amazonia": # Umidade tende a cair no inverno fora da Amazônia
        umidade = max(10, umidade - np.random.uniform(10,30))


    densidade = round(np.random.uniform(10, 5000 if 'urbano' in regiao["usos_solo"] else 500),1) # Densidade maior em áreas com potencial urbano
    altitude = round(np.random.uniform(regiao["altitude_range"][0], regiao["altitude_range"][1]), 1)
    declive = round(np.random.uniform(regiao["declive_range"][0], regiao["declive_range"][1]), 1)
    dist_agua = round(np.random.uniform(0.1, 80 if regiao_escolhida_nome == "Norte_Amazonia" else 30), 1) # Amazônia pode ter pontos mais distantes de cursos d'água mapeados, mas é rica em água

    solo = random.choice(regiao["tipos_solo"])
    uso = random.choice(regiao["usos_solo"])

    acess = random.random()
    if uso == 'urbano': acessibilidade = 'alta' if random.random() < 0.8 else 'media'
    elif uso == 'floresta' or regiao_escolhida_nome == "Norte_Amazonia": acessibilidade = 'baixa' if random.random() < 0.7 else 'media'
    else: acessibilidade = random.choice(['baixa', 'media', 'alta'])
        
    sismo = np.random.poisson(0.1) # Sismos são raros e de baixa magnitude no Brasil

    # 5. Lógica de Risco e Ocorrência (mantida, mas será influenciada pelas features regionais)
    # A probabilidade de ocorrência é ajustada para ser um pouco mais frequente para fins de dataset sintético
    ocorrencia = 0
    risco = 'BAIXO'
    tipo = random.choice(regiao["eventos_comuns"]) # Começa com um evento comum da região

    # Incêndio
    if (temperatura > 32 and umidade < 40 and chuva < 30 and (uso in ['floresta', 'pastagem', 'rural'] or solo == 'arenoso') and
        (estacao == 'inverno' or regiao_escolhida_nome in ["Nordeste_Sertao", "CentroOeste_Cerrado_Pantanal"])):
        tipo = 'incêndio'
        risco = 'ALTO' if temperatura > 35 and umidade < 25 else 'MÉDIO'
        ocorrencia = 1 if random.random() < (0.6 if risco == 'ALTO' else 0.3) else 0
    
    # Deslizamento
    elif chuva > 150 and declive > 25 and solo in ['argiloso', 'argiloso_seco', 'organico']:
        tipo = 'deslizamento'
        risco = 'ALTO' if chuva > 200 and declive > 35 else 'MÉDIO'
        ocorrencia = 1 if random.random() < (0.5 if risco == 'ALTO' else 0.25) else 0

    # Seca
    elif (chuva < 20 and temperatura > 30 and umidade < 50 and 
          regiao_escolhida_nome == "Nordeste_Sertao" and estacao != 'verao'): # Mais específico para Sertão
        tipo = 'seca'
        risco = 'ALTO' if chuva < 10 and temperatura > 35 else 'MÉDIO'
        ocorrencia = 1 if random.random() < (0.7 if risco == 'ALTO' else 0.4) else 0
        if ocorrencia == 1: umidade = round(np.random.uniform(10,30),1) # Se seca ocorre, umidade é baixa

    # Inundação
    elif (chuva > 180 and (dist_agua < 5 or uso == 'area_umida' or altitude < 100) and 
          declive < 15 and uso != 'floresta_densa'): # Floresta densa pode mitigar
        tipo = 'inundação'
        risco = 'ALTO' if chuva > 250 and dist_agua < 2 else 'MÉDIO'
        ocorrencia = 1 if random.random() < (0.55 if risco == 'ALTO' else 0.3) else 0

    # Vendaval (menos dependente de solo/uso, mais de condições atmosféricas gerais)
    elif temperatura > 25 and umidade > 60 and random.random() < 0.1: # Condição genérica para instabilidade
        if 'vendaval' in regiao["eventos_comuns"]: # Apenas se comum na região
            tipo = 'vendaval'
            risco = 'MÉDIO' # Vendavais fortes são mais complexos de simular assim
            ocorrencia = 1 if random.random() < 0.15 else 0 # Mais raros

    # Se nenhuma condição específica forte foi atendida, mas ainda pode haver um risco menor
    if ocorrencia == 0 and random.random() < 0.1: # Pequena chance de um evento de baixo risco ocorrer
        risco = 'BAIXO'
        ocorrencia = 1 if random.random() < 0.05 else 0 # Ocorrência rara para baixo risco
        # tipo já foi escolhido de regiao["eventos_comuns"]
    elif ocorrencia == 0: # Se ainda não ocorreu, e não é baixo risco aleatório
        risco = random.choice(['BAIXO', 'MUITO BAIXO']) if risco != 'MÉDIO' else 'MÉDIO' # Evita sobrescrever um MÉDIO já definido
        if risco == 'MUITO BAIXO': tipo = "nenhum" # Adicionar um tipo "nenhum"

    if tipo == "nenhum": ocorrencia = 0; risco = "MUITO BAIXO"


    return [
        f'Z_{id_zona:04d}_{regiao_escolhida_nome[:3].upper()}', # ID da Zona com código da região
        lat, lon, data.strftime('%Y-%m-%d'),
        chuva, temperatura, umidade, densidade, altitude, declive,
        dist_agua, solo, uso, acessibilidade, sismo,
        tipo, ocorrencia, risco, mes, estacao,
        regiao_escolhida_nome # Adicionar a região como uma feature
    ]

# Gera o dataset
dados = [gerar_linha_por_regiao(i) for i in range(1, n_amostras + 1)]

# Define colunas
colunas = [
    'id_zona', 'lat', 'lon', 'data_referencia',
    'chuva_mm', 'temperatura_media', 'umidade_media', 'densidade_populacional',
    'altitude_m', 'declividade_terreno', 'proximidade_curso_agua_km',
    'tipo_solo', 'uso_solo', 'acessibilidade', 'atividade_sismica_recente',
    'tipo_evento', 'ocorrencia_evento', 'risco_previsto', # 'risco_previsto' é o 'risco' gerado
    'mes', 'estacao',
    'regiao_brasil' # Nova coluna
]

# Adicionar 'nenhum' à lista de tipos de evento se for usado
if any("nenhum" in linha for linha in dados):
    tipos_evento_geral = list(set(tipos_evento_geral + ['nenhum']))


df = pd.DataFrame(dados, columns=colunas)

# Salva
df.to_csv('dataset_risco_desastres_brasil_v3.csv', index=False)
print("✅ Novo dataset 'dataset_risco_desastres_brasil_v3.csv' gerado com lógica regional.")
print(f"\nTipos de evento gerados: {df['tipo_evento'].unique()}")
print(f"Regiões geradas: {df['regiao_brasil'].value_counts(normalize=True)}")
print(f"Ocorrências de evento: \n{df['ocorrencia_evento'].value_counts(normalize=True)}")