#!/usr/bin/env python3
# encoding: utf-8
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


assert (len(sys.argv) >= 2), 'Debe haber un argumento'

matrix_train = pd.read_csv('/gaa/home/edcastil/datos/20131231.mdata.control_resolucion.csv', index_col=0)
matrix_val = pd.read_csv('/gaa/home/edcastil/datos/20141231.mdata.control_resolucion.csv', index_col=0)

prod_train = pd.read_csv('/gaa/home/edcastil/datos/Prod_2013_trihorario.csv', index_col=0)
prod_val = pd.read_csv('/gaa/home/edcastil/datos/Prod_2014_trihorario.csv', index_col=0)

variables = list(matrix_train.columns)
target = prod_train.columns[0]
n_dimensiones = len(variables)

x_train = matrix_train.values
x_val = matrix_val.values

y_train = prod_train.values
y_val = prod_val.values

#Escalado de datos
scaler = MinMaxScaler()
x_train_escalado = scaler.fit_transform(x_train)
x_val_escalado = scaler.transform(x_val)

'''SVR parametrizado'''
parametros = eval(sys.argv[1])
type(parametros)
i = parametros[0]
j = parametros[1]
k = parametros[2]

svr = SVR(C=i, gamma=k, epsilon=j, kernel='rbf', shrinking = True, tol = 1.e-6)
svr.fit(x_train_escalado,y_train.ravel())
y_pred = svr.predict(x_val_escalado)
mae = mean_absolute_error(y_val, y_pred)

lista_resultados = [parametros, mae]
#lista_predicciones = [y_val, y_pred]
nombre = '/gaa/home/edcastil/scripts/resultados_control/resultados_svr_control_' + str(parametros) + '.txt'
f = open(nombre, 'w')
f.write(str(lista_resultados) + '\n')
f.close()