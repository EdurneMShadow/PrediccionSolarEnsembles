#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import numpy as np
import pandas as pd
import DataMatrix_NWP as dm
import datetime
import pickle
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

import DataMatrix_NWP as dm


        
#Carga de datos + conjuntos entrenamiento, validación y test
assert (len(sys.argv) >= 2), 'Debe haber un argumento'
assert (len(sys.argv[1]) == 3), 'El modelo necesita tres parámetros'

matrix_train = (dm.DataMatrix(datetime.datetime(2013,12,31), 
'/gaa/home/data/solar_ecmwf/', '/gaa/home/data/solar_ecmwf/', ifexists = True, 
model='deterministic', suffix='.det_noacc_vmodule'))

matrix_val = (dm.DataMatrix(datetime.datetime(2014,12,31), 
'/gaa/home/data/solar_ecmwf/', '/gaa/home/data/solar_ecmwf/', ifexists = True, 
model='deterministic', suffix='.det_noacc_vmodule'))

matrix_test = (dm.DataMatrix(datetime.datetime(2015,12,31), 
'/gaa/home/data/solar_ecmwf/', '/gaa/home/data/solar_ecmwf/', ifexists = True, 
model='deterministic', suffix='.det_noacc_vmodule'))

prod_train = pd.read_csv('/gaa/home/edcastil/datos/Prod_2013.csv', index_col=0)
prod_val = pd.read_csv('/gaa/home/edcastil/datos/Prod_2014.csv', index_col=0)
prod_test = pd.read_csv('/gaa/home/edcastil/datos/Prod_2015.csv', index_col=0)

variables = list(matrix_train.dataMatrix.columns)
target = prod_train.columns[0]
#print('Variables: ' + str(variables))
#print('Target: ' + str(target))

n_dimensiones = len(variables)
x_train = matrix_train.dataMatrix.values
x_val = matrix_val.dataMatrix.values
x_test = matrix_test.dataMatrix.values
y_train = prod_train.values
y_val = prod_val.values
y_test = prod_test.values

#Escalado de datos
scaler = StandardScaler()
x_train_escalado = scaler.fit_transform(x_train)
x_val_escalado = scaler.fit_transform(x_val)
x_test_escalado = scaler.fit_transform(x_test)

'''SVR parametrizado'''
parametros = sys.argv[1]
i = parametros[0]
j = parametros[1]
k = parametros[2]

svr = SVR(C=i, gamma=k, epsilon=j, kernel='rbf', shrinking = True, tol = 1.e-6)
svr.fit(x_train_escalado,y_train)
y_pred = svr.predict(x_val_escalado)
mae = mean_absolute_error(y_val, y_pred)

lista_resultados = [parametros, mae]
f = open('resultados_svr.txt', 'w')
f.write(lista_resultados)
f.close()




















