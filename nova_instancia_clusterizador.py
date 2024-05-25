import pandas as pd
from pickle import load
import numpy as np
pd.set_option('display.max_columns', None)


bank_clusters_kmeans = load(open("arquivos/bank_clusters.pkl", "rb"))
normalizador = load(open("arquivos/bank_normalizador.pkl", "rb"))
colunas_categoricas = load(open("arquivos/colunas_categoricas.pkl", "rb"))

teste_instancia = [56,"blue-collar","married","basic.4y","no","no","no","telephone","may","mon",261,1,999,0,"nonexistent",1.1,93.994,-36.4,4.857,5191,"no"]

# Criar DataFrame a partir da lista de teste
df_teste = pd.DataFrame([teste_instancia], columns=['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'dayofweek', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'])
#normalização de dados categoricos
dados_categoricos_normalizados = pd.get_dummies(data=df_teste[['job','marital','education','default','housing','loan', 'contact', 'month', 'dayofweek', 'poutcome', 'y']], dtype=int)
#normalização de dados numéricos
dados_numericos = df_teste.drop(columns=['job','marital','education','default','housing','loan', 'contact', 'month', 'dayofweek', 'poutcome', 'y'])
dados_numericos_normalizados = normalizador.transform(dados_numericos)
dados_numericos_normalizados = pd.DataFrame(data= dados_numericos_normalizados, columns= ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])
# print(dados_numericos_normalizados)
#instancia dos dados das colunas
dados_categoricos = pd.DataFrame(columns=colunas_categoricas)
print(len(dados_categoricos.columns))
#junção dos dados normalizados com os dados das colunas
dados_completos = pd.concat([dados_categoricos, dados_categoricos_normalizados], axis=0)
dados_completos = dados_completos.where(pd.notna(dados_completos), other=0)
dados_completos = dados_numericos_normalizados.join(dados_completos, how='left')
# dados_completos.to_excel('aaaa.xlsx')

# Exibir o DataFrame resultante

centroide = bank_clusters_kmeans.cluster_centers_[bank_clusters_kmeans.predict(dados_completos)]

# inverse_transform dados_numericos
# from dummies pra categoricos
dados_normalizados_final_legiveis = normalizador.inverse_transform(dados_numericos_normalizados)
dados_categoricos_legiveis = pd.from_dummies(dados_categoricos_normalizados, sep='_')
dados_normalizados_final_legiveis = pd.DataFrame(data= dados_normalizados_final_legiveis, columns= ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']).join(dados_categoricos_legiveis)

print(dados_normalizados_final_legiveis)