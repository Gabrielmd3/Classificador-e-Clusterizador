import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
from sklearn.model_selection import cross_validate, cross_val_score
dados = pd.read_csv('dados/bank-additional-full.csv', sep = ';')

dados_atributos = dados.drop(columns=['job','marital','education','default','housing','loan', 'contact', 'month', 'dayofweek', 'poutcome', 'y'])
dados_classes = dados[['y']]
dados_classes_para_normalizar = dados[['job','marital','education','default','housing','loan', 'contact', 'month', 'dayofweek', 'poutcome']]

# print(dados_atributos)
# print(dados_classes)

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler()

bank_normalizador = normalizador.fit(dados_atributos)
from pickle import dump
dump(bank_normalizador, open('arquivos/bank_normalizador_classificador.pwl', 'wb'))

####################################################################################################################################

dados_atributos_normalizados = bank_normalizador.fit_transform(dados_atributos)
dados_atributos_normalizados = pd.DataFrame(data= dados_atributos_normalizados, columns= dados_atributos.columns)
dados_categoricos_normalizados = pd.get_dummies(data= dados_classes_para_normalizar, dtype='int16')
colunas_categoricas = dados_categoricos_normalizados.columns
dump(colunas_categoricas, open("arquivos/colunas_categoricas_classificadores.pkl", "wb"))
# exit()
# dados_normalizados_final = dados_atributos_normalizados.join(dados_categoricos_normalizados, how='left')

dados_normalizados_final_legivel = bank_normalizador.inverse_transform(dados_atributos_normalizados)
dados_normalizados_final_legivel = pd.DataFrame(data= dados_normalizados_final_legivel, columns= dados_atributos_normalizados.columns).join(dados_categoricos_normalizados)

dados_finais = pd.DataFrame(dados_normalizados_final_legivel, columns=dados_normalizados_final_legivel.columns)
dados_finais = dados_finais.join(dados_classes, how='left')

print(len(dados_finais))

####################################################################################################################################
# print('Freq das classes original: ', dados_classes.value_counts())

dados_finais_train = dados_finais.sample(n=677)
dados_atributos = dados_finais_train.drop(columns=['y'])
dados_classes = dados_finais_train['y']

resampler = SMOTE()
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)

print('Frequência de classes após balanceamento')

classes_count = Counter(dados_classes_b)

dados_atributos_b = pd.DataFrame(dados_atributos_b)
dados_classes_b = pd.DataFrame(dados_classes_b)

# print(dados_atributos_b)

####################################################################################################################################


####################################################################################################################################
atributos_train, atributos_test, classes_train, classes_test = train_test_split(dados_atributos_b, dados_classes_b)

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2],
    'min_samples_split': [5, 19],
    'min_weight_fraction_leaf': [0.2, 0.3],
    'max_features': ['sqrt', 'log2'],
}


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

gridsearch = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=10)
gridsearch.fit(atributos_train, classes_train)

print("Melhores Parâmetros: ", gridsearch.best_params_)
print("Melhor Score: ", gridsearch.best_score_)

tree = DecisionTreeClassifier(criterion= gridsearch.best_params_['criterion'], splitter= gridsearch.best_params_['splitter'], max_depth= gridsearch.best_params_['max_depth'],
                              min_samples_split= gridsearch.best_params_['min_samples_split'], min_weight_fraction_leaf = gridsearch.best_params_['min_weight_fraction_leaf'],
                                max_features= gridsearch.best_params_['max_features'])


dados_finais_train = dados_finais
dados_atributos = dados_finais_train.drop(columns=['y'])
dados_classes = dados_finais_train['y']
dados_atributos_b = pd.DataFrame(dados_atributos_b)
dados_classes_b = pd.DataFrame(dados_classes_b)
atributos_train, atributos_test, classes_train, classes_test = train_test_split(dados_atributos_b, dados_classes_b)


bank_tree = tree.fit(atributos_train, classes_train) # Este treinamento não faz sentido quando usamos CrossValidation

Classe_test_predict = bank_tree.predict(atributos_test)

# exit()
# # Comparar as classes inferidas no teste com as classes preservadas no split
# i = 0
# for i in range(0, len(classes_test)):
#     print(classes_test.iloc[i][0], ' - ', Classe_test_predict[i])

####################################################################################################################################

scoring = ['precision_macro', 'recall_macro']
scores_cross = cross_validate(tree, dados_atributos_b, dados_classes_b, cv = 10, scoring = scoring)

print(scores_cross['test_precision_macro'].mean())
print(scores_cross['test_recall_macro'].mean())
score_cross_val = cross_val_score(tree, dados_atributos_b, dados_classes_b, cv = 10)
print(score_cross_val.mean(), ' - ', score_cross_val.std())

bank_tree = tree.fit(dados_atributos_b, dados_classes_b)

dump(bank_tree, open('arquivos/bank_tree_model_cross.pwl', 'wb'))


####################################################################################################################################
bank_tree_cross = tree.fit(dados_atributos_b, dados_classes_b)

dump(bank_tree_cross, open('arquivos/bank_tree_cross.pkl', 'wb'))
# exit()