# Documentação do Projeto: Modelos de Desastres e Risco (ARYA IA)

Este documento detalha a estrutura e o funcionamento do projeto ARYA IA, uma aplicação web em Flask desenvolvida para classificação e previsão de risco de desastres naturais.

## Visão Geral do Projeto
O projeto ARYA IA é uma solução para a análise e previsão de desastres naturais, utilizando modelos de Machine Learning e uma interface web interativa. O objetivo é fornecer ferramentas para classificar tipos de desastres e prever seus níveis de risco, visualizando essas informações geograficamente.

## Funcionalidades Principais
- **Interface Web Intuitiva**: Desenvolvida com Flask e HTML/CSS, oferece uma navegação simples para acessar as funcionalidades de classificação e predição.
- **Classificador de Desastres**: Implementa um modelo de aprendizado de máquina para categorizar o tipo de desastre com base em dados de entrada. Os resultados são apresentados em um mapa interativo.
- **Preditor de Risco**: Contém um modelo de aprendizado de máquina que estima o nível de risco associado a um evento, também visualizado em um mapa geográfico.
- **Visualização Geoespacial (Plotly)**: Utiliza a biblioteca Plotly para criar mapas dinâmicos que exibem as localizações dos desastres, suas classificações e níveis de risco.
- **Exportação de Dados MongoDB para CSV**: Inclui scripts para extrair dados brutos de coleções MongoDB, transformando-os em arquivos CSV, que são utilizados para treinamento e inferência dos modelos de IA.
- **Pipeline de Pré-processamento de Dados**: Os modelos de IA incorporam etapas de pré-processamento, como tratamento de valores ausentes, codificação de variáveis categóricas e engenharia de features, garantindo a qualidade e adequação dos dados para os algoritmos.

## Estrutura do Projeto
A organização do projeto segue a seguinte estrutura de diretórios:

```plaintext
├── app.py  
├── requirements.txt  
├── Static/  
│   └── style.css  
├── Templates/  
│   ├── classificador_map.html  
│   ├── index.html  
│   └── preditor_map.html  
├── Data/  
│   ├── CSV/  
│   │   ├── desastres_naturais_20250608_141114.csv  
│   │   └── relatorios_desastres_20250608_141218.csv  
│   └── MongoDB/  
│       ├── desastres_naturais.py  
│       └── relatorios_desastres.py  
└── src/  
    ├── Classificador_de_Desastres/  
    │   └── classificador.py  
    └── Preditor_de_Risco/  
        └── preditor.py
```

- **app.py**: O arquivo principal da aplicação Flask, responsável pelas rotas, lógica de carregamento dos modelos e renderização das páginas web.
- **requirements.txt**: Lista todas as bibliotecas Python necessárias para o projeto.
- **Static/**: Contém arquivos estáticos da aplicação, como CSS.
  - **style.css**: Folha de estilos para a interface web.
- **Templates/**: Armazena os templates HTML usados pelo Flask.
  - **classificador_map.html**: Template para exibir o mapa do classificador de desastres.
  - **index.html**: Página inicial da aplicação.
  - **preditor_map.html**: Template para exibir o mapa do preditor de risco.
- **Data/**: Contém os scripts de exportação de dados e os arquivos CSV resultantes.
  - **CSV/**: Armazena os datasets em formato CSV.
    - **desastres_naturais_20250608_141114.csv**: Dados de desastres naturais.
    - **relatorios_desastres_20250608_141218.csv**: Dados de relatórios de desastres.
  - **MongoDB/**: Contém scripts para interagir com o MongoDB.
    - **desastres_naturais.py**: Script para exportar a coleção `desastres_naturais` do MongoDB para CSV.
    - **relatorios_desastres.py**: Script para exportar a coleção `relatorios_desastres` do MongoDB para CSV.
- **src/**: Contém os módulos Python dos modelos de IA.
  - **Classificador_de_Desastres/**: Módulo do classificador.
    - **classificador.py**: Lógica para o modelo de classificação de desastres.
  - **Preditor_de_Risco/**: Módulo do preditor.
    - **preditor.py**: Lógica para o modelo de predição de risco.

## Análise Detalhada do Código
Esta seção explora o propósito e a lógica principal de cada arquivo Python relevante.

### app.py
Este é o coração da aplicação Flask, responsável por rotear as requisições web e orquestrar a interação com os modelos de IA.

- **Importações**: Importa Flask, `render_template` para lidar com as rotas e templates HTML, e as classes dos modelos de IA (`ClassificadorDeDesastres` e `PreditorDeRisco`) de seus respectivos módulos.
- **Inicialização do Flask**: `app = Flask(__name__)` cria a instância da aplicação Flask.
- **Carregamento dos Modelos**:
  - Instancia `ClassificadorDeDesastres()` e `PreditorDeRisco()`.
  - Chama `classificador.carregar_dados()` e `preditor.carregar_dados()` para que os modelos carreguem seus respectivos datasets de `Data/CSV/`.
  - Chama `classificador.treinar_modelo()` e `preditor.treinar_modelo()` para treinar os modelos. Este treinamento ocorre na inicialização da aplicação, o que é adequado para demonstrações, mas em produção, os modelos treinados seriam geralmente carregados de arquivos persistidos (ex: `.pkl`).
- **Rotas (`@app.route`)**:
  - `/` (Home): Renderiza `index.html`, a página inicial que oferece as opções de classificação e predição.
  - `/classificador`:
    - Chama `classificador.prever()` para obter as previsões do modelo classificador.
    - Chama `classificador.criar_mapa()` para gerar o mapa Plotly com as classificações.
    - Renderiza `classificador_map.html`, passando o JSON do mapa gerado para ser incorporado na página.
  - `/preditor`:
    - Chama `preditor.prever()` para obter as previsões do modelo preditor.
    - Chama `preditor.criar_mapa()` para gerar o mapa Plotly com os níveis de risco.
    - Renderiza `preditor_map.html`, passando o JSON do mapa gerado.

### src/Classificador_de_Desastres/classificador.py
Este módulo encapsula toda a lógica para o modelo de classificação de desastres.

- **Classe `ClassificadorDeDesastres`**:
  - `__init__(self)`: Construtor da classe. Inicializa `self.dados` como `None`, `self.modelo` como `None` e `self.colunas_modelo` como uma lista vazia, que será populada com as colunas usadas para treinamento.
  - `carregar_dados(self)`:
    - Carrega o arquivo `desastres_naturais_20250608_141114.csv` em um DataFrame Pandas.
    - Realiza pré-processamento dos dados:
      - Trata valores ausentes (`fillna(0)` para colunas numéricas como `precipitacao (mm)`, `temperatura_c`, `umidade_percentual`).
      - Converte a coluna 'ocorreu' para tipo booleano.
      - Cria novas features (Engenharia de Features):
        - `mes`: Extrai o mês da coluna 'data'.
        - `dia_da_semana`: Extrai o dia da semana da coluna 'data'.
        - `estacao_cod`: Mapeia a 'estacao_do_ano' para valores numéricos.
        - `regiao_cod`: Mapeia a 'regiao' para valores numéricos.
      - Define `self.colunas_modelo` com as features que serão usadas no treinamento.
  - `treinar_modelo(self)`:
    - Define as colunas de features (X) e a coluna alvo (y - 'tipo_evento').
    - Divide os dados em conjuntos de treinamento e teste (`train_test_split`).
    - **Treinamento do Modelo**: Utiliza um `RandomForestClassifier` para treinar o modelo.
      - **Por que RandomForestClassifier?**: Este modelo foi escolhido para a tarefa de classificação devido à sua robustez e capacidade de lidar com dados complexos. É um algoritmo de ensemble que constrói múltiplas árvores de decisão durante o treinamento e produz a classe que é o modo das classes (classificação) das árvores individuais.
      - **Vantagens**:
        - Alta Acurácia: Geralmente tem um bom desempenho em termos de acurácia.
        - Resistência a Overfitting: A construção de múltiplas árvores e a agregação dos resultados ajudam a reduzir o overfitting, um problema comum em árvores de decisão únicas.
        - Lida bem com dados não-lineares e interações entre features: Não assume uma relação linear entre as variáveis, tornando-o adequado para dados de desastres que podem ter padrões complexos.
        - Importância de Features: Permite avaliar a importância de cada feature na decisão, o que pode ser útil para entender os fatores que mais contribuem para a classificação de um desastre.
        - Manuseio de Dados Ausentes: Embora o pré-processamento trate NaNs, o RandomForest é relativamente tolerante a dados ausentes, o que pode ser uma vantagem em datasets reais.
    - **Avaliação (Opcional)**: Contém blocos comentados para cálculo de métricas de avaliação do modelo (acurácia, precisão, recall, F1-score).
    - Armazena o modelo treinado em `self.modelo`.
  - `prever(self)`:
    - Garanti que o modelo e os dados estejam carregados.
    - Usa `self.modelo.predict(self.dados[self.colunas_modelo])` para fazer as previsões.
    - Adiciona a coluna `tipo_previsto` ao DataFrame `self.dados`.
    - Retorna o DataFrame com as previsões.
  - `criar_mapa(self)`:
    - Verifica se os dados estão disponíveis.
    - Cria um mapa de dispersão geoespacial usando `plotly.express.scatter_mapbox`.
    - **Parâmetros do Mapa**: Define as coordenadas (`lat='latitude'`, `lon='longitude'`), colore os pontos pelo `tipo_previsto`, exibe `nome_atual` ao passar o mouse, e define o zoom e estilo do mapa.
    - Converte o objeto figura Plotly para JSON para ser incorporado no template HTML.
    - Retorna o JSON do mapa.

### src/Preditor_de_Risco/preditor.py
Este módulo é dedicado à lógica do modelo de predição de risco de desastres.

- **Classe `PreditorDeRisco`**:
  - `__init__(self)`: Construtor, inicializa `self.dados`, `self.modelo` e `self.colunas_modelo`.
  - `carregar_dados(self)`:
    - Carrega o arquivo `relatorios_desastres_20250608_141218.csv` em um DataFrame Pandas.
    - Realiza pré-processamento de dados semelhante ao classificador: Trata valores ausentes, converte 'ocorreu' para booleano, cria features de tempo, e cria features categóricas codificadas.
    - Define `self.colunas_modelo` para as features do preditor.
  - `treinar_modelo(self)`:
    - Define as colunas de features (X) e a coluna alvo (y - 'nivel_de_risco').
    - Divide os dados em treinamento e teste.
    - **Treinamento do Modelo**: Utiliza um `RandomForestRegressor` para treinar o modelo de regressão.
      - **Por que RandomForestRegressor?**: Similar ao classificador, o RandomForest Regressor é uma escolha robusta para problemas de regressão (onde a saída é um valor contínuo, como o nível de risco). Ele opera construindo múltiplas árvores de decisão e calculando a média das previsões das árvores individuais para chegar a uma previsão final.
      - **Vantagens**:
        - Alta Acurácia: Conhecido por seu bom desempenho em problemas de regressão.
        - Resistência a Overfitting: A natureza de ensemble mitiga o risco de overfitting, que é crucial para garantir que o modelo generalize bem para novos dados.
        - Lida com Relações Não-Lineares: Capaz de modelar relações complexas entre as features e o nível de risco, que dificilmente seriam lineares na realidade de desastres naturais.
        - Menos Sensível a Ruído e Outliers: A agregação de múltiplas árvores torna o modelo menos sensível a pontos de dados ruidosos ou outliers.
        - Importância de Features: Assim como o classificador, permite identificar quais fatores (chuva, temperatura, etc.) são mais influentes na previsão do nível de risco.
    - **Avaliação (Opcional)**: Blocos comentados para avaliação do regressor (e.g., `mean_squared_error`).
    - Armazena o modelo treinado.
  - `prever(self)`:
    - Garanti que o modelo e os dados estejam carregados.
    - Usa `self.modelo.predict(self.dados[self.colunas_modelo])` para gerar as previsões de risco.
    - Arredonda as previsões para o número inteiro mais próximo e as adiciona como `risco_previsto`.
    - Retorna o DataFrame com as previsões.
  - `criar_mapa(self)`:
    - Cria um mapa de dispersão geoespacial Plotly semelhante ao classificador.
    - **Parâmetros do Mapa**: Define as coordenadas, colore e dimensiona os pontos com base no `risco_previsto`, e exibe `nome_atual` ao passar o mouse.
    - Converte a figura para JSON.
    - Retorna o JSON do mapa.

### Data/MongoDB/desastres_naturais.py
Este script é responsável por conectar ao MongoDB e exportar a coleção `desastres_naturais` para um arquivo CSV.

- **Importações**: `MongoClient` para conexão com MongoDB e `pandas` para manipulação de DataFrames.
- **Conexão ao MongoDB**: Estabelece uma conexão com o servidor MongoDB (padrão: `mongodb://localhost:27017/`).
- **Seleção do Banco de Dados e Coleção**: Conecta-se ao banco de dados `arya_ia` e à coleção `desastres_naturais`.
- **Consulta e Exportação**: Recupera todos os documentos da coleção, converte-os em um DataFrame Pandas e salva o DataFrame em um arquivo CSV na pasta `Data/CSV/`. O nome do arquivo inclui um timestamp.
- **Fechamento da Conexão**: Garante que a conexão com o MongoDB seja fechada.

### Data/MongoDB/relatorios_desastres.py
Este script é similar a `desastres_naturais.py`, mas focado na exportação da coleção `relatorios_desastres`.

A lógica é idêntica a `desastres_naturais.py`, mas interage com a coleção `relatorios_desastres` e salva o CSV correspondente na mesma pasta `Data/CSV/`.

## Dados
O projeto utiliza os seguintes arquivos CSV como fonte de dados para os modelos de IA:

- `Data/CSV/desastres_naturais_20250608_141114.csv`: Contém informações sobre desastres naturais, usadas pelo `ClassificadorDeDesastres`.
- `Data/CSV/relatorios_desastres_20250608_141218.csv`: Contém dados de relatórios de desastres, usados pelo `PreditorDeRisco`.

Os scripts de exportação do MongoDB geram arquivos CSV com nomes baseados em timestamp, portanto, os nomes exatos dos arquivos CSV podem variar.

## Dependências
As dependências do projeto estão listadas no arquivo `requirements.txt`:

## 🔗 Links Úteis

* [Link do GitHub](https://github.com/ARYA-GS/arya-ia)
* [Link do Youtube](https://youtu.be/nX3npID70W4)

---

## 👥 Integrantes

| Nome | RM |
| :--- | :--- |
| José Neto | 553844 |
| Vitor Cruz | 553621 |
| Keven Ike | 553215 |
