#!/usr/bin/env python3
# encoding: utf-8
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

ruta_imagen = 'Imagenes/Control/'
ruta_resultados = 'Resultados/Ensembles/'

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
f = open(ruta_resultados + 'comparaciones_svr_test_control_ens.pkl', 'rb')
prod_test = pd.read_csv('Produccion/Prod_2015_ens.csv', index_col=0)
lista_comparacion = pickle.load(f)
y_test = lista_comparacion[0]
y_pred_control = lista_comparacion[1]
#y_pred = pickle.load(f)
f.close()

y = np.arange(250)
a = y 
#plt.figure( figsize = (7, 6) )
#plt.title('y_test vs y_pred')
#plt.xlabel('y_test')
#plt.ylabel('y_pred')
#_ = plt.plot(prod_test, y_pred, 'o')
##_ = plt.plot(a,y)
#_ = plt.plot(prod_test,prod_test)
#plt.savefig(ruta_imagen + 'comparacion.png')

plt.figure( figsize = (7, 6) )
plt.title('y_test vs y_pred')
plt.xlabel('y_test')
plt.ylabel('y_pred')
_ = plt.plot(prod_test, y_pred_control, 'o')
#_ = plt.plot(a,y)
_ = plt.plot(prod_test,prod_test)
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

''''=======================================================================
                                Ensembles
    =======================================================================
''''
df = pd.DataFrame()
cols = []
for i in range(50):
    f = open(ruta_resultados + 'comparaciones_svr_ensemble_desacc_' +str(i) + '.pkl', 'rb')
    y_pred = pickle.load(f)
    df_aux = pd.DataFrame(y_pred)
    df = pd.concat([df, y_pred], axis=1)
    cols.append('y_pred_' + str(i))
data = pd.read_csv('/gaa/home/edcastil/datos/20151231.mdata.ens_desacc_0.csv')

index = data.index
df.columns = cols
df.index = index
df.to_csv('/gaa/home/edcastil/scripts/y_pred_ensembles.csv')

#Hacer medias por filas para obtener un dataFrame de una única columna
y_pred = pd.read_csv('/gaa/home/edcastil/datos/y_pred_ensembles.csv')
df_mean = y_pred.mean(axis=1)
df_median = y_pred.median(axis = 1)

df_mean = pd.DataFrame(df_mean)
df_median = pd.DataFrame(df_median)

df_mean.index = y_pred.index
df_median.index = y_pred.index

df_mean.columns = 'y_pred_ens_mean'
df_median.columns = 'y_pred_ens_median'

df_mean.to_csv('/gaa/home/edcastil/datos/y_pred_ens_mean.csv')
df_median.to_csv('/gaa/home/edcastil/datos/y_pred_ens_median.csv')

#Calcular nuevo mae para la media y la mediana
prod_test = pd.read_csv('/gaa/home/edcastil/datos/Prod_2015_ens.csv', index_col=0)
y_test = prod_test.values
mae_mean = mean_absolute_error(y_test, df_mean)
mae_median = mean_absolute_error(y_test, df_median)

print('MAE_mean: ' + str(mae_mean))
print('MAE_median: ' + str(mae_median))

    
    
    
    
    
    
    
    
    

