# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:07:02 2017

@author: Edurne
"""

#get_ipython().magic('matplotlib inline')
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer


#Carga de datos
df = pd.read_csv('w_e.csv', sep=',')

print("nFilas: %d\tnColumnas: %d\n" % (df.shape[0], df.shape[1]) )
print("Columnas:\n", np.array(df.columns))

l_features = list(df.columns[2:])
target = df.columns[1]

n_dimensiones = len(l_features)
x = df[l_features].values
y = df[target].values


#Conjunto de entrenamiento y test
conjuntos = ShuffleSplit(test_size = 0.25)

#Escalado de datos
scaler = StandardScaler()
x_escalado = scaler.fit_transform(x)

'''Modelo SVR predefinido'''
#Modelo SVR
svr = SVR(C=1., gamma = 1/n_dimensiones, epsilon = y.std()/10., kernel = 'rbf', shrinking = False, tol = 1.e-6)
t_0 = time.time()
svr.fit(x_escalado, y)
print('Tiempo de train: ', time.time()-t_0)
print('Número de vectores de soporte: ', svr.support_.shape[0])

#MAE
scores_mae = - cross_val_score(svr, x_escalado, y, cv=conjuntos, scoring = 'neg_mean_absolute_error', n_jobs=8)
print('cv mae mean: ', scores_mae.mean())

'''SVR parametrizado'''
lista_C = [10.**k for k in range (0,5)]
lista_gamma = list(np.array([2.**k for k in range(-2, 4)])/n_dimensiones)
lista_epsilon = list(y.std() * np.array([2.**k for k in range(-6, -2)]))

parametros = {'C': lista_C, 'gamma': lista_gamma, 'epsilon': lista_epsilon}
print('Número de parámetros: ', len(lista_C)*len(lista_gamma)*len(lista_epsilon))

svr_parametrizado = SVR(kernel = 'rbf', shrinking=True, tol=1.e-3)

conjuntos = ShuffleSplit(n_splits=5, test_size = 0.25)
buscador = (GridSearchCV(svr_parametrizado, param_grid=parametros, cv = conjuntos, 
                         scoring='neg_mean_absolute_error', n_jobs=8, verbose=5))

t_0 = time.time()
buscador.fit(x_escalado,y)
print('Tiempo de búsqueda: ', time.time() - t_0)

with open('svr_parametrizado.txt' , 'wb') as handle:
    pickle.dump(buscador, handle, protocol= pickle.HIGHEST_PROTOCOL)
    
#Mejores parámetros
with open('svr_parametrizado.txt', 'rb') as handle :
    buscador = pickle.load(handle)
best_C = buscador.best_params_['C']
best_gamma = buscador.best_params_['gamma']
best_epsilon = buscador.best_params_['epsilon']
print('Mejor C: ', best_C, ' Mejor gamma: ', best_gamma, ' Mejor epsilon: ', best_epsilon)

#Modelo con los mejores parámetros
svr_best = SVR(C=best_C, gamma=best_gamma, epsilon=best_epsilon, kernel='rbf', shrinking=True, tol=1.e-3)

#Evaluacion con kfold cross-validation + mae
kf = KFold(10, shuffle=True, random_state=0)
scores_mae_parametrizado = (- cross_val_score(svr_best, x_escalado,y, cv=kf, 
                                              scoring='neg_mean_absolute_error', n_jobs=8, verbose=5))
print('Media Scores MAE SVR parametrizado: ', scores_mae_parametrizado.mean())

















