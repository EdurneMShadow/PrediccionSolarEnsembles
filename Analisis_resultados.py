#!/usr/bin/env python3
# encoding: utf-8
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

ruta_imagen = 'Imagenes/Control/'
ruta_resultados = 'Resultados/Control/'

''''==========================================================================
                                PLOTS PARÁMETROS
    ==========================================================================
''''
c = []
e = []
g = []
mae = []

for linea in open(ruta_resultados + 'resultados_svr_control.txt'):
    linea = eval(linea)
    parametros = linea[0]
    c.append(parametros[0])
    e.append(parametros[1])
    g.append(parametros[2])
    mae.append(linea[1])
    
plt.title('C vs MAE')
plt.xlabel('C')
plt.ylabel('MAE')
plt.xscale('log')
_ = plt.plot(c, mae, 'o')
plt.savefig(ruta_imagen + 'parametros_c.png')

plt.title('Epsilon vs MAE')
plt.xlabel('Epsilon')
plt.ylabel('MAE')
plt.xscale('log')
_ = plt.plot(e, mae, 'o')
plt.savefig(ruta_imagen + 'parametros_e.png')

plt.title('Gamma vs MAE')
plt.xlabel('Gamma')
plt.ylabel('MAE')
plt.xscale('log')
_ = plt.plot(g, mae, 'o')
plt.savefig(ruta_imagen + 'parametros_g.png')

''''=======================================================================
                                PLOT COMPARACIÓN
    =======================================================================
''''
#Prediccion vs original
f = open(ruta_resultados + 'comparaciones_svr_test_control.pkl', 'rb')
lista_comparacion = pickle.load(f)
y_test = lista_comparacion[0]
y_pred = lista_comparacion[1]
f.close()

y = np.arange(250)
a = y 
plt.figure( figsize = (7, 6) )
plt.title('y_test vs y_pred')
plt.xlabel('y_test')
plt.ylabel('y_pred')
_ = plt.plot(y_test, y_pred, 'o')
_ = plt.plot(a,y)
plt.savefig(ruta_imagen + 'comparacion.png')

''''=====================================================================
                            HISTOGRAMAS
   =======================================================================
''''
prod_train = pd.read_csv('Produccion/Prod_2015_resolucion.csv', index_col=0)
plt.title('Distribución de la producción en 2015 sin noche')
plt.xlabel('% de producción')
plt.ylabel('n_horas')
plt.xticks(np.arange(0,81,5)) 
_ = plt.hist(prod_train.values, bins = 41)
plt.savefig('Imagenes/produccion2015_resolucion_nuevo.png')

''''=======================================================================
                            DATAFRAME PARÁMETROS
    =======================================================================
''''
    
c1 = pd.DataFrame(c, columns=['C'])
c2 = pd.DataFrame(e, columns=['Epsilon'])
c3 = pd.DataFrame(g, columns=['Gamma'])
c4 = pd.DataFrame(mae, columns=['MAE'])

parametros = pd.concat([c1,c2,c3,c4], axis=1)

par = parametros.sort(['MAE'], ascending=[True])
index = np.arange(240)
par.index = index

nombre = ruta_resultados + 'parametros_df.pkl'
pickle.dump(par, open(nombre, 'wb' ))

top10 = np.arange(10)
t10 = par.loc[top10]
new_index = np.arange(1,11)
t10.index = new_index
print(t10.to_latex())


''''=======================================================================
                                INTERPOLACIÓN
    =======================================================================
''''
y_test = pd.read_csv('Produccion/Prod_2015_resolucion.csv', index_col=0)
y_interpolada = pd.read_csv(ruta_resultados + 'y_interpolada_control.csv', index_col=0)
h = np.arange(24)

horas = np.arange(365*16)
y = np.arange(90)
a = y 
plt.figure( figsize = (7, 6) )
plt.title('y_pred vs y_test')
plt.xlabel('y_original')
plt.ylabel('y_interpolada')
_ = plt.plot(y_test.values,y_interpolada.values, 'o')
_ = plt.plot(a,y)
plt.savefig(ruta_imagen + 'comparacion_interpolacion_control.png')


''''=======================================================================
                                CS_mean
    =======================================================================
''''
cs = pd.read_csv('cs_2015_mean.csv', index_col=0)
inicio = datetime.datetime(2015,10,1)
fin = datetime.datetime(2015,10,8)
index = lib.obtener_indice_luminoso(inicio, fin)
octubre = cs.loc[index]

y = np.arange(128)
plt.title('CS_mean')
plt.xlabel('horas')
plt.ylabel('radiación_cs')
_ = plt.plot(y, octubre)
plt.savefig('Imagenes/cs_mean_octubre.png')

