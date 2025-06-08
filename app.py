from flask import Flask, render_template, jsonify
import json
import plotly
import os
import sys


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

CLASSIFICADOR_MODEL_PATH = "C:/Users/zenet/OneDrive/Desktop/ARYA_IA_GS/arya-ia/src/Classificador_de_Desastres/classificador.py"
PREDITOR_MODEL_PATH = "C:/Users/zenet/OneDrive/Desktop/ARYA_IA_GS/arya-ia/src/Preditor_de_Risco/preditor.py"


sys.path.append(os.path.join(BASE_DIR, 'src'))


try:

    from Classificador_de_Desastres.classificador import main_classificador
    from Preditor_de_Risco.preditor import main_preditor
except ImportError as e:
    print(f"Erro ao importar módulos dos modelos: {e}")
    print(f"Verifique se os arquivos '{CLASSIFICADOR_MODEL_PATH}' e '{PREDITOR_MODEL_PATH}' existem e se não há erros de sintaxe neles.")
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