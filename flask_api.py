
#Importando bibliotecas
from flask import Flask, request
import pickle
import pandas as pd
from pycaret.classification import load_model, predict_model

#variavel instanciando o flask
app = Flask( __name__)

#load model 
#model = pickle.load(open('C:\\Users\\adeni\Desktop\\final_model.pkl', 'rb'))
model = load_model('C:\\Users\\adeni\Desktop\\final_model')



#criando uma rota para a api e selecionando o método POST, indicando
#que vamos inserir uma informação para que a api retorne algo.
#se fosse GET, indicaria que só vamos capturar dados
@app.route('/predict', methods=['POST'])
#toda rota executa uma função ao ser acessada
def predict():
    #pega os dados no formato json (pq toda comunicação entre apis é em json)
    test_json = request.get_json()
    #COLETA OS DADOS
    #se existe o arquivo test_json
    if test_json:
        #ok, tem o arquivo json, se ele for um dicionário, quer dizer que só tem uma linha
        if isinstance(test_json, dict):
            #então cria um dataframe com o index começando em 0
            raw_data = pd.DataFrame(test_json, index=[0])
        #se não for um dicionário, tem mais de uma linha
        else:
            #então pega o json, transforma em um df informando que as colunas são as chaves da primeira linha do json
            raw_data = pd.DataFrame(test_json, columns=test_json[0].keys())
    #PREDICTION
    pred = predict_model(model, data=raw_data) #Execução do modelo de Classificação
    pred = pred['Label']
    #inclui coluna no df com predições
    raw_data['prediction'] = pred

    #transforma novamente para json
    return raw_data.to_json(orient='records')



#start flask
if __name__ == '__main__':
    #start flask
    app.run(host='0.0.0.0', port='5000')