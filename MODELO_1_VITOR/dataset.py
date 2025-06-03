import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime
from shapely.geometry import Point
import geopandas as gpd

faker = Faker('pt_BR')
n = 300

# Categorias
source_types = ["Drone", "App Usuário", "Equipe de Campo"]
observation_types = ["Alagamento", "Deslizamento", "Incêndio", "Estrutura Danificada", "Pessoas Isoladas"]
severity_levels = ["Baixa", "Média", "Alta", "Crítica"]
damage_levels = ["Nenhum", "Leve", "Moderado", "Severo"]
accessibility_levels = ["Fácil", "Difícil", "Bloqueado"]
risk_classes = ["Monitorar", "Suporte Necessário", "Atenção Urgente", "Risco Imediato"]

# Carrega o shape dos países e filtra o Brasil
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
brasil = world[world.name == 'Brazil']

# Função para gerar ponto válido dentro do Brasil
def gerar_ponto_valido():
    while True:
        lon = random.uniform(-74, -34)
        lat = random.uniform(-34, 5.3)
        ponto = Point(lon, lat)
        if brasil.contains(ponto).bool():
            return round(lat, 6), round(lon, 6)

# Função de classificação de risco
def classificar_risco(severidade, atingidos, dano, acesso):
    score = 0
    if severidade == "Crítica": score += 3
    elif severidade == "Alta": score += 2
    elif severidade == "Média": score += 1
    if atingidos > 100: score += 2
    elif atingidos > 50: score += 1
    if dano == "Severo": score += 2
    elif dano == "Moderado": score += 1
    if acesso == "Bloqueado": score += 2
    elif acesso == "Difícil": score += 1
    if score >= 7: return "Risco Imediato"
    elif score >= 5: return "Atenção Urgente"
    elif score >= 3: return "Suporte Necessário"
    else: return "Monitorar"

# Geração dos dados
data = []

for i in range(1, n + 1):
    timestamp = faker.date_time_between(start_date='-30d', end_date='now')
    latitude, longitude = gerar_ponto_valido()
    source = random.choice(source_types)
    obs_type = random.choice(observation_types)
    severity = random.choices(severity_levels, weights=[0.2, 0.3, 0.3, 0.2])[0]
    num_affected = np.random.randint(0, 200)
    damage = random.choice(damage_levels)
    access = random.choice(accessibility_levels)
    notes = faker.sentence(nb_words=random.randint(6, 12))
    risk = classificar_risco(severity, num_affected, damage, access)

    data.append([
        i, timestamp, latitude, longitude, source, obs_type, severity,
        num_affected, damage, access, notes, risk
    ])

# Criação do DataFrame
df = pd.DataFrame(data, columns=[
    "report_id", "timestamp", "latitude", "longitude", "source_type",
    "observation_type", "severity_reported", "num_affected_estimate",
    "infrastructure_damage", "accessibility", "additional_notes", "AREA_RISK_CLASSIFICATION"
])

# Salvar CSV
df.to_csv("desastres_reports_brasil.csv", index=False, encoding='utf-8')
print("Dataset 'desastres_reports_brasil.csv' gerado com sucesso com coordenadas dentro do Brasil!")
