#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
##### leer csvs

df_2013 = pd.read_csv('/gaa/home/edcastil/datos/20131231.mdata.control_desagregado_resolucion.csv', index_col = 0)
df_2014 = pd.read_csv('/gaa/home/edcastil/datos/20141231.mdata.control_desagregado_resolucion.csv', index_col = 0)
df_2015 = pd.read_csv('/gaa/home/edcastil/datos/20151231.mdata.control_desagregado_resolucion.csv', index_col = 0)

##### columnas SSRD
l_cols = df_2013.columns
l_cols_SSRD = []

for col in l_cols:
    if col.find('SSRD') > 0:
        l_cols_SSRD.append(col)

##### means via numpy
mean_2013 = df_2013[l_cols_SSRD].values.mean(axis=0)
max_2013  = df_2013[l_cols_SSRD].values.max(axis=0)

mean_2014 = df_2014[l_cols_SSRD].values.mean(axis=0)
max_2014  = df_2014[l_cols_SSRD].values.max(axis=0)

mean_2015 = df_2015[l_cols_SSRD].values.mean(axis=0)
max_2015  = df_2015[l_cols_SSRD].values.max(axis=0)

print("max_2013/max_2014: ", np.log(mean_2013.max()/mean_2014.max()), 
      "\nmax_2013/max_2015: ", mean_2013.max()/mean_2015.max(),
      "\nmax_2014/max_2015: ", mean_2014.max()/mean_2015.max())

#==================================================================

#df_2013 = pd.read_csv('/gaa/home/edcastil/datos/20131231.mdata.det_trihorario.csv', index_col = 0)
#df_2014 = pd.read_csv('/gaa/home/edcastil/datos/20141231.mdata.control_resolucion.csv', index_col = 0)
#df_2015 = pd.read_csv('/gaa/home/edcastil/datos/20151231.mdata.control_resolucion.csv', index_col = 0)
#prod_train = pd.read_csv('/gaa/home/edcastil/datos/Prod_2013_trihorario_des.csv', index_col=0)
#prod_val = pd.read_csv('/gaa/home/edcastil/datos/Prod_2014_trihorario_des.csv', index_col=0)
#prod_test = pd.read_csv('/gaa/home/edcastil/datos/Prod_2015_trihorario_des.csv', index_col=0)
#
#x_train = df_2013.values
#x_val = df_2014.values
#x_test = df_2015.values
#y_train = prod_train.values
#y_val = prod_val.values
#y_test = prod_test.values
#
#scaler = MinMaxScaler()
#
#x_train_escalado = scaler.fit_transform(x_train)
#x_val_escalado = scaler.transform(x_val)
#x_test_escalado = scaler.transform(x_test)
#
#errores = {}
#for linea in open('resultados_svr_trihorario_nparametros.txt'):
#    linea = eval(linea)
#    parametros = linea[0]
#    mae = linea[1]
#    errores[parametros] = mae
#
#minimo_mae = 100
#for i in errores.keys():
#    if errores[i] < minimo_mae:
#        minimo_mae = errores[i]
#        clave = i
#
#svr = SVR(C=clave[0], gamma=clave[2], epsilon=clave[1], kernel='rbf', shrinking = True, tol = 1.e-6)
#svr.fit(x_train_escalado,y_train.ravel())
#
#y_train_pred = svr.predict(x_train_escalado)
#mae = mean_absolute_error(y_train, y_train_pred)
#print('Mae_train: ' + str(mae))
#
#y_val_pred = svr.predict(x_val_escalado)
#mae = mean_absolute_error(y_val, y_val_pred)
#print('Mae_val: ' + str(mae))
#
#y_test_pred = svr.predict(x_test_escalado)
#mae = mean_absolute_error(y_test, y_test_pred)
#print('Mae_test: ' + str(mae))
#
##=============================================================================
##leer 2013 det
#matrix_train = pd.read_csv('20131231.mdata.det_trihorario.csv', index_col=0)
#
##leer 2014, 2015 control
#matrix_val  = pd.read_csv('20141231.mdata.control_desagregado_resolucion.csv', index_col=0)
#matrix_test = pd.read_csv('20151231.mdata.control_desagregado_resolucion.csv', index_col=0)
#
###### columnas SSRD
l_cols = matrix_train.columns
l_cols_SSRD = []
for col in l_cols:
    if col.find('SSRD') > 0:                           
        l_cols_SSRD.append(col)

###### max, means
print("calculando max y medias de columnas SSRD ...")
ssrd_mean_2013 = matrix_train[l_cols_SSRD].values.mean(axis=0)
ssrd_max_2013  = matrix_train[l_cols_SSRD].values.max(axis=0)
ssrd_mean_2014 = matrix_val[l_cols_SSRD].values.mean(axis=0)
ssrd_max_2014  = matrix_val[l_cols_SSRD].values.max(axis=0)

l_cols_SSRD_reversed = l_cols_SSRD[ : : -1]
ssrd_mean_2014_reversed = matrix_val[l_cols_SSRD_reversed].values.mean(axis=0)
ssrd_max_2014_reversed  = matrix_val[l_cols_SSRD_reversed].values.max(axis=0)

ssrd_mean_2015 = matrix_test[l_cols_SSRD].values.mean(axis=0)
ssrd_max_2015  = matrix_test[l_cols_SSRD].values.max(axis=0)
#
plt.title('2013 vs 2014_reversed')
plt.plot(ssrd_mean_2013, ssrd_mean_2014_reversed, '*')
plt.savefig('2013vs2014_reversed.png')
plt.show()
#
plt.title('2013 vs 2014')
plt.plot(ssrd_mean_2013, ssrd_mean_2014, '*')
#plt.savefig('2013vs2015.png')
plt.savefig('Imagenes/2013vs2014.png')
plt.show()
#
plt.title('2014_reversed vs 2015')
plt.plot(ssrd_mean_2014_reversed, ssrd_mean_2015, '*')
plt.savefig('2014reversedvs2015.png')
plt.show()


print("max_2013/max_2014: ", ssrd_mean_2013.max()/ssrd_mean_2014.max(), 
      "\nmax_2013/max_2015: ", ssrd_mean_2013.max()/ssrd_mean_2015.max(),
      "\nmax_2014/max_2015: ", ssrd_mean_2014.max()/ssrd_mean_2015.max())



