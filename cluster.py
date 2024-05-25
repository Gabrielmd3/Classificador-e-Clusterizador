import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from pickle import dump
import warnings
warnings.filterwarnings("ignore")


bank = pd.read_csv('dados/bank-additional-full.csv', sep = ';')
#Normalizar os dados categoricos
dados_categoricos = bank[['job','marital','education','default','housing','loan', 'contact', 'month', 'dayofweek', 'poutcome', 'y']]
dados_numericos = bank.drop(columns=['job','marital','education','default','housing','loan', 'contact', 'month', 'dayofweek', 'poutcome', 'y'])
dados_categoricos_normalizados = pd.get_dummies(data= dados_categoricos, dtype='int16')
colunas_categoricas = dados_categoricos_normalizados.columns
dump(colunas_categoricas, open("arquivos/colunas_categoricas.pkl", "wb"))

#normalizar dados numericos
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_numericos)
dump(modelo_normalizador, open('arquivos/bank_normalizador.pkl','wb'))

#criar um dataframe dos dados normalizados
dados_numericos_normalizados = modelo_normalizador.fit_transform(dados_numericos)
dados_numericos_normalizados = pd.DataFrame(data= dados_numericos_normalizados, columns= ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed'])
dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos_normalizados, how='left')

#determinar um numero ótimo de clusters
distortions = []
K =range (1,100)
for i in K:
    bank_kmeans_model = KMeans(n_clusters=i, n_init='auto', random_state=42).fit(dados_normalizados_final)
    distortions.append(
      sum(np.min(
          cdist(dados_normalizados_final,bank_kmeans_model.cluster_centers_, 'euclidean'), axis =1)/dados_normalizados_final.shape[0])
      )

# Calcular o número ótimo de clusters
x0 = K[0]
y0 = distortions[0]
xn = K[len(K) -1]
yn = distortions[len(distortions)-1]

# Iterar nos pontos gerados durante os treinamentos preliminares
distancias = []
for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs((yn-y0) * x - (xn-x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn-y0)**2 + (xn - x0)**2)
    distancias.append(numerador/denominador)

n_clusters_otimos = K[distancias.index(np.max(distancias))]

# Treinar o modelo definitivo
bank_kmeans_model = KMeans(n_clusters = n_clusters_otimos, random_state=42).fit(dados_normalizados_final)
dump(bank_kmeans_model, open('arquivos/bank_clusters.pkl', 'wb'))