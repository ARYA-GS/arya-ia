from flask import Flask, render_template, jsonify
import json
import plotly
import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

import importlib.util

def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Não foi possível encontrar a especificação para o módulo {module_name} em {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    classificador_file_path = os.path.join(BASE_DIR, 'src', 'Classificador_de_Desastres', 'classificador.py')
    preditor_file_path = os.path.join(BASE_DIR, 'src', 'Preditor_de_Risco', 'preditor.py')

    classificador_module = load_module_from_path('classificador_module', classificador_file_path)
    preditor_module = load_module_from_path('preditor_module', preditor_file_path)

    main_classificador = classificador_module.main_classificador
    main_preditor = preditor_module.main_preditor

except ImportError as e:
    print(f"Erro ao importar módulos dos modelos: {e}")
    print(f"Tentando carregar Classificador de: {classificador_file_path}")
    print(f"Tentando carregar Preditor de: {preditor_file_path}")
    print("Verifique se os arquivos existem, se os caminhos estão corretos e se não há erros de sintaxe nos modelos.")
    sys.exit(1)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classificador_map')
def classificador_map():
    fig = main_classificador() 

    if fig:
        graph_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('classificador_map.html', graph_json=graph_json, title="Mapa do Classificador de Desastres")
    else:
        return "Erro ao gerar o mapa do Classificador de Desastres.", 500


@app.route('/preditor_map')
def preditor_map():
    fig = main_preditor() 
    if fig:
        graph_json = json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('preditor_map.html', graph_json=graph_json, title="Mapa do Preditor de Risco")
    else:
        return "Erro ao gerar o mapa do Preditor de Risco.", 500

if __name__ == '__main__':
    app.run(debug=True)