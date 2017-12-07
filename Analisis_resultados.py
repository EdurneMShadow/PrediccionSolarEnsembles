#!/usr/bin/env python3
# encoding: utf-8
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

c = []
epsilon = []
gamma = []
mae = []
for linea in open('Resultados/Determinista_trihorario/resultados_svr_trihorario.txt'):
    linea = eval(linea)
    parametros = linea[0]
    c.append(parametros[0])
    epsilon.append(parametros[1])
    gamma.append(parametros[2])
    mae.append(linea[1])
    
x = set(c)
#MAE vs parametros
plt.figure( figsize = (15, 12) )
plt.subplot(2, 2, 1)
plt.title('C vs MAE')
plt.xlabel('C')
plt.ylabel('MAE')
#plt.yticks(np.arange(max(set(mae))), mae) 
plt.xticks(np.array([1.0, 10.0, 100.0, 1000.0, 10000.0]), np.array([1, 10, 100, 1000, 10000]))
plt.xscale('log')
_ = plt.plot(c, mae, 'o')
plt.savefig('Imagenes/Determinista_trihorario/parametros_c.png')

plt.subplot(2, 2, 2)
plt.title('Epsilon vs MAE')
plt.xlabel('Epsilon')
plt.ylabel('MAE')
plt.xticks(np.array([0.4168831566623316, 0.8337663133246632, 1.6675326266493264, 3.335065253298653]))
_ = plt.plot(epsilon, mae, 'o')
plt.savefig('Imagenes/Determinista_horario/parametros_e.png')

plt.subplot(2, 2, 3)
plt.title('Gamma vs MAE')
plt.xlabel('Gamma')
plt.ylabel('MAE')
#plt.xticks(np.array([3.2063614210593817e-06,6.4127228421187634e-06, 1.2825445684237527e-05, 2.5650891368475054e-05, 5.130178273695011e-05, 0.00010260356547390021]), rotation=50) 
_ = plt.plot(gamma, mae, 'o')  
plt.savefig('Imagenes/Determinista_trihorario/parametros_g.png')

#Prediccion vs original
f = open('Resultados/Determinista_trihorario/comparaciones_svr_test_trihorario.pkl', 'rb')
lista_comparacion = pickle.load(f)
y_test = lista_comparacion[0]
y_pred = lista_comparacion[1]
f.close()

h = np.arange(24)
y = np.arange(250)
a = y 
plt.figure( figsize = (7, 6) )
plt.title('y_pred vs y_test')
plt.xlabel('y_pred')
plt.ylabel('y_test')
_ = plt.plot(y_pred, y_test, 'o')
_ = plt.plot(a,y)
plt.savefig('Imagenes/Determinista_trihorario/comparacion.png')

#Histogramas
prod_train = pd.read_csv('Produccion/Prod_2015_resolucion.csv', index_col=0)
plt.title('Distribución de la producción en 2015 sin noche')
plt.xlabel('% de producción')
plt.ylabel('n_horas')
plt.xticks(np.arange(0,81,5)) 
_ = plt.hist(prod_train.values, bins = 41)
plt.savefig('Imagenes/produccion2015_resolucion_nuevo.png')

#DataFrame MAE
c = []
e = []
g = []
mae = []

for linea in open('Resultados/Resultados_horario_resolucion_repe/resultados_svr_resolucion_repe.txt'):
    linea = eval(linea)
    parametros = linea[0]
    c.append(parametros[0])
    e.append(parametros[1])
    g.append(parametros[2])
    mae.append(linea[1])
    
c1 = pd.DataFrame(c, columns=['C'])
c2 = pd.DataFrame(e, columns=['Epsilon'])
c3 = pd.DataFrame(g, columns=['Gamma'])
c4 = pd.DataFrame(mae, columns=['MAE'])

parametros = pd.concat([c1,c2,c3,c4], axis=1)

par = parametros.sort(['MAE'], ascending=[True])
index = np.arange(120)
par.index = index

nombre = 'Resultados/Resultados_horario_resolucion_repe/parametros_horario_repe.pkl'
pickle.dump(par, open(nombre, 'wb' ))

a = np.arange(1,11)
par.loc[a]


#Interpolación vs original
y_test = pd.read_csv('Produccion/Prod_2015_resolucion.csv', index_col=0)
y_interpolada = pd.read_csv('Resultados/Determinista_trihorario/y_interpolada_trihorario01.csv', index_col=0)
h = np.arange(24)

horas = np.arange(365*16)
y = np.arange(100)
a = y 
plt.figure( figsize = (7, 6) )
plt.title('y_pred vs y_test')
plt.xlabel('y_interpolada')
plt.ylabel('y_original')
_ = plt.plot(y_interpolada.values, y_test.values, 'o')
_ = plt.plot(a,y)
plt.savefig('Imagenes/Determinista_trihorario/comparacion_interpolacion.png')
