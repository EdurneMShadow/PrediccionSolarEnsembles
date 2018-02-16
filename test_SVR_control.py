#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import pandas as pd
import glob
import pickle

matrix_train = pd.read_csv('/gaa/home/edcastil/datos/20131231.mdata.control_desagregado_resolucion.csv', index_col=0)
matrix_test = pd.read_csv('/gaa/home/edcastil/datos/20151231.mdata.control_nuevo_desagregado.csv', index_col=0)

prod_train = pd.read_csv('/gaa/home/edcastil/datos/Prod_2013_trihorario.csv', index_col=0)
prod_test = pd.read_csv('/gaa/home/edcastil/datos/Prod_2015_trihorario.csv', index_col=0)

for i in glob.glob("/gaa/home/edcastil/scripts/resultados_control/*"):
    f = open(i)
    out = open('resultados_svr_control.txt', 'a')
    out.write(f.read())
    f.close()
out.close()
    
x_train = matrix_train.values
x_test = matrix_test.values
y_train = prod_train.values
y_test = prod_test.values


scaler = MinMaxScaler()
x_train_escalado = scaler.fit_transform(x_train)
x_test_escalado = scaler.transform(x_test)

errores = {}
for linea in open('resultados_svr_control.txt'):
    linea = eval(linea)
    parametros = linea[0]
    mae = linea[1]
    errores[parametros] = mae

minimo_mae = 100
for i in errores.keys():
    if errores[i] < minimo_mae:
        minimo_mae = errores[i]
        clave = i
print('Parametros con menor MAE: ' + str(clave) + ' MAE: ' + str(minimo_mae))

svr = SVR(C=clave[0], gamma=clave[2], epsilon=clave[1], kernel='rbf', shrinking = True, tol = 1.e-6)
svr.fit(x_train_escalado,y_train.ravel())
#y_train_pred = svr.predict(x_train_escalado)
y_pred = svr.predict(x_test_escalado)
mae = mean_absolute_error(y_test, y_pred)

lista_predicciones = [y_train, y_pred]
nombre = 'comparaciones_svr_test_control.pkl'
pickle.dump(lista_predicciones, open(nombre, 'wb' ))

nombre = 'resultados_test_control.txt'
f = open(nombre, 'w')
f.write(str(clave) + '\n')
f.write('Error de test: ' + str(mae) + '\n')
f.close()