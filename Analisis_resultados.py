#!/usr/bin/env python3
# encoding: utf-8
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import matplotlib.pyplot as plt
import pickle
import numpy as np

c = []
epsilon = []
gamma = []
mae = []
for linea in open('Resultados/Determinista_horario_resolucion/resultados_svr_resolucion.txt'):
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
plt.savefig('Imagenes/Determinista_horario/parametros_c.png')

plt.subplot(2, 2, 2)
plt.title('Epsilon vs MAE')
plt.xlabel('Epsilon')
plt.ylabel('MAE')
plt.xticks(np.array([0.42860043682455884,0.8572008736491177, 1.7144017472982354, 3.4288034945964707]))
_ = plt.plot(epsilon, mae, 'o')
plt.savefig('Imagenes/Determinista_horario/parametros_e.png')

plt.subplot(2, 2, 3)
plt.title('Gamma vs MAE')
plt.xlabel('Gamma')
plt.ylabel('MAE')
#plt.xticks(np.array([3.2063614210593817e-06,6.4127228421187634e-06, 1.2825445684237527e-05, 2.5650891368475054e-05, 5.130178273695011e-05, 0.00010260356547390021]), rotation=50)
_ = plt.plot(gamma, mae, 'o')  
plt.savefig('Imagenes/Determinista_horario/parametros_g.png')

#Prediccion vs original
f = open('Resultados/Determinista_horario_resolucion/comparaciones_svr_test.pkl', 'rb')
lista_comparacion = pickle.load(f)  
y_test = lista_comparacion[0]
y_pred = lista_comparacion[1]
f.close()

h = np.arange(24)
y = np.arange(92)
a = y
plt.figure( figsize = (7, 6) )
plt.title('y_pred vs y_test')
plt.xlabel('y_pred')
plt.ylabel('y_test')
_ = plt.plot(y_pred, y_test, 'o')
_ = plt.plot(a,y)
plt.savefig('Imagenes/Determinista_horario/comparacion.png')