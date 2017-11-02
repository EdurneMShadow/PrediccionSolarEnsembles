#!/usr/bin/env python3
# encoding: utf-8
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import matplotlib.pyplot as plt
import pickle

c = []
epsilon = []
gamma = []
mae = []
for linea in open('resultados_svr_resolucion.txt'):
    linea = eval(linea)
    parametros = linea[0]
    c.append(parametros[0])
    epsilon.append(parametros[1])
    gamma.append(parametros[2])
    mae.append(linea[1])
    
    
#MAE vs parametros
plt.figure( figsize = (15, 18) )
plt.subplot(3, 1, 1)
plt.title('C vs MAE')
plt.xlabel('C')
plt.ylabel('MAE')
#plt.xticks(y)
_ = plt.plot(mae, c, 'o')

plt.subplot(3, 1, 2)
plt.title('Epsilon vs MAE')
plt.xlabel('Epsilon')
plt.ylabel('MAE')
#plt.xticks(y)
_ = plt.plot(mae, epsilon, 'o')

plt.subplot(3, 1, 3)
plt.title('Gamma vs MAE')
plt.xlabel('Gamma')
plt.ylabel('MAE')
#plt.xticks(y)
_ = plt.plot(mae, gamma, 'o')  


#Prediccion vs original
f = open('comparaciones_svr_test.pkl', 'rb')
lista_comparacion = pickle.load(f)  
y_test = lista_comparacion[0]
y_pred = lista_comparacion[1]

plt.figure( figsize = (15, 18) )
_ = plt.plot(y_pred, y_test, 'o')
