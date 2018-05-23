#!/usr/bin/env python3
# encoding: utf-8
"""
Created on Sun Feb  4 22:33:29 2018

@author: Edurne
"""
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import pickle

assert (len(sys.argv) >= 3), 'Debe haber dos argumentos'
fichero = sys.argv[1]
n_ensemble = sys.argv[2]


#matrix_train = pd.read_csv('/gaa/home/edcastil/datos/20131231.mdata.control_ens.csv', index_col=0) #desagregado_resolucion
matrix_train = pd.read_csv('/gaa/home/edcastil/datos/control/20131231.mdata.control_desagregado_resolucion.csv', index_col=0)
matrix_test = pd.read_csv(fichero, index_col=0)

prod_train = pd.read_csv('/gaa/home/edcastil/datos/Prod_2013_trihorario.csv', index_col=0)
prod_test = pd.read_csv('/gaa/home/edcastil/datos/Prod_2015_trihorario.csv', index_col=0)

x_train = matrix_train.values
x_test = matrix_test.values
y_train = prod_train.values
y_test = prod_test.values


scaler = MinMaxScaler()
x_train_escalado = scaler.fit_transform(x_train)
x_test_escalado = scaler.transform(x_test)

parametros = (100.0, 0.15473252272286195, 0.0009578544061302681)

svr = SVR(C=parametros[0], gamma=parametros[2], epsilon=parametros[1], kernel='rbf', shrinking = True, tol = 1.e-6)
svr.fit(x_train_escalado,y_train.ravel())
y_pred = svr.predict(x_test_escalado)
mae = mean_absolute_error(y_test, y_pred)

#lista_predicciones = [y_train, y_pred]
nombre = 'comparaciones_svr_ensemble_' + n_ensemble + '.pkl'
pickle.dump(y_pred, open(nombre, 'wb' ))

nombre = 'resultados_ensemble_' + n_ensemble + '.txt'
f = open(nombre, 'w')
f.write(str(parametros) + '\n')
f.write('Error de test: ' + str(mae) + '\n')
f.close()