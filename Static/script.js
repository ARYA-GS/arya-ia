// static/script.js

document.addEventListener('DOMContentLoaded', () => {
    const classifierBtn = document.getElementById('runClassifierBtn');
    const predictorBtn = document.getElementById('runPredictorBtn');
    const chartDiv = document.getElementById('plotlyChart');
    const loader = document.getElementById('loader');
    const infoBox = document.getElementById('info');

    // Função genérica para chamar a API e renderizar o gráfico
    const runModel = async (endpoint) => {
        // Limpa o estado anterior
        chartDiv.innerHTML = '';
        infoBox.style.display = 'none';
        infoBox.innerText = '';
        loader.style.display = 'block'; // Mostra o loader

        try {
            // Chama a API do Flask
            const response = await fetch(endpoint);
            if (!response.ok) {
                throw new Error(`Erro na API: ${response.statusText}`);
            }
            const data = await response.json();

            let plotData;
            
            // O endpoint do preditor retorna um objeto com 'plot' e 'accuracy'
            if (data.plot) {
                plotData = data.plot;
                // Mostra a acurácia
                infoBox.innerText = `Acurácia do Modelo: ${data.accuracy}`;
                infoBox.style.display = 'block';
            } else {
                plotData = data; // O classificador retorna o plot diretamente
            }

            // Usa Plotly.js para renderizar o gráfico
            Plotly.newPlot('plotlyChart', plotData.data, plotData.layout);

        } catch (error) {
            console.error('Falha ao rodar o modelo:', error);
            chartDiv.innerHTML = `<p style="color: red; text-align: center;">Ocorreu um erro ao gerar o gráfico. Verifique o console do servidor.</p>`;
        } finally {
            loader.style.display = 'none'; // Esconde o loader
        }
    };

    // Adiciona os eventos aos botões
    classifierBtn.addEventListener('click', () => runModel('/run_classifier'));
    predictorBtn.addEventListener('click', () => runModel('/run_predictor'));
});