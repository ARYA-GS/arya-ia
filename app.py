# app.py

import json
from flask import Flask, render_template, jsonify
import plotly.graph_objects as go

# Importe as funções que acabamos de criar
from src.Classificador_de_Desastres.train import run_classification
from src.Preditor_de_Risco.train import run_prediction

# Inicializa o aplicativo Flask
app = Flask(__name__)

@app.route('/')
def index():
    """ Rota principal que renderiza a página HTML. """
    return render_template('index.html')

@app.route('/run_classifier')
def run_classifier_endpoint():
    """ Endpoint da API para rodar o modelo de classificação. """
    try:
        # Executa o modelo e obtém a figura
        fig = run_classification()
        # Converte a figura para JSON
        graph_json = json.loads(fig.to_json())
        return jsonify(graph_json)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run_predictor')
def run_predictor_endpoint():
    """ Endpoint da API para rodar o modelo de predição. """
    try:
        # Executa o modelo e obtém a figura e a acurácia
        fig, accuracy = run_prediction()
        # Converte a figura para JSON
        graph_json = json.loads(fig.to_json())
        # Adiciona a acurácia ao JSON de resposta
        response_data = {
            "plot": graph_json,
            "accuracy": f"{accuracy:.2%}"
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Roda o servidor em modo de debug
    app.run(debug=True)