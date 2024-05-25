from pickle import load
import pandas as pd
import numpy as np

nova_instancia = ["56", "housemaid.", "married", "basic.9y", "yes", "no", "yes", "telephone", "may", "mon", 45, 1, 987, 1, "nonexistent", 2.2, 90.981, -30.4, 4.857, 5191]

# Normalizar a nova instância
# Abrir o modelo normalizador salvo antes do treinamento
normalizador = load(open('arquivos/bank_normalizador_classificador.pwl', 'rb'))
colunas_categoricas = load(open("arquivos/colunas_categoricas_classificadores.pkl", "rb"))
# exit()
# Carregar os nomes das colunas categóricas

# Lista de teste
teste_instancia = ["56", "admin.", "married", "basic.9y", "yes", "no", "yes", "telephone", "may", "mon", 45, 1, 987, 1, "nonexistent", 2.2, 90.981, -30.4, 4.857, 5191]

pd.set_option('display.max_columns', None)

# Criar DataFrame a partir da lista de teste
df_teste = pd.DataFrame([teste_instancia], columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'dayofweek', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])

# Converter colunas categóricas em dummy variables
dados_categoricos_normalizados = pd.get_dummies(data=df_teste[['job','marital','education','default','housing','loan', 'contact', 'month', 'dayofweek', 'poutcome']], dtype=int)
dados_numericos = df_teste.drop(columns=['job','marital','education','default','housing','loan', 'contact', 'month', 'dayofweek', 'poutcome'])
dados_numericos_normalizados = normalizador.transform(dados_numericos)

dados_numericos_normalizados = pd.DataFrame(data= dados_numericos_normalizados, columns= ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])
# print(dados_numericos_normalizados)
dados_categoricos = pd.DataFrame(columns=colunas_categoricas)

# Concatenar os DataFrames
dados_completos = pd.concat([dados_categoricos, dados_categoricos_normalizados], axis=0)

# Substituir os valores NaN por pd.NA
dados_completos = dados_completos.where(pd.notna(dados_completos), other=0)
dados_completos = dados_numericos_normalizados.join(dados_completos, how='left')

# Classificar a nova instância
# Abrir o modelo classificador salvo anteriormente
bank_classificador = load(open('arquivos/bank_tree_model_cross.pwl', 'rb'))

# Classificar
resultado = bank_classificador.predict(dados_completos)

dist_proba = bank_classificador.predict_proba(dados_completos)

# print("Classe: " + resultado)
# print(dist_proba)

indice = np.argmax(dist_proba[0])
classe_predita = bank_classificador.classes_[indice]
score = dist_proba[0][indice]
print("Classificado como: ", classe_predita, "Score: ", str(score))
print(np.argmax(dist_proba[0]))
print(bank_classificador.classes_)