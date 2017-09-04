# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:26:54 2017

@author: Edurne
"""

import pandas as pd
import numpy as np
import desagregar as lib
import datetime
import matplotlib.pyplot as plt

def mae_prueba(original, interpolado):
    print("MAE: ", abs(original - interpolado).mean())
    print("RMAE: ", abs(original - interpolado).mean()/original.mean())

def plot_diagonal(original, interpolado, nombre):
    plt.plot(original, interpolado,'.')
    plt.savefig(nombre + '.pdf')
    plt.show()
    
def plot_comparacion(original, interpolado, nombre):
    y = np.arange(0,24,1)
    _ = plt.plot(y,original, label = 'original')
    _ = plt.plot(y, interpolado, label = 'interpolado')
    plt.xticks(y)
    plt.legend(loc = 'best')
    plt.savefig(nombre + '.pdf')
    plt.show()



        
m_trihoraria = pd.read_csv('./Prueba_interpolacion/matriz_punto_trihoraria_viento.csv', index_col = 0)
m_original = pd.read_csv('./Prueba_interpolacion/matriz_punto_original_viento.csv', index_col = 0)
 
dates = pd.date_range('20150322','20150323',freq='3H')[:-1]
index3 = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])

marzo_trihorario = m_trihoraria.loc[index3]

dates = pd.date_range('20150322','20150323',freq='1H')[:-1]
index1 = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])

tags = ['TCC','T2M','U10','V10','v10']

for i, tag in enumerate(tags):
    m_interpolada_horaria = lib.interpolacion_cubica(marzo_trihorario, index1, tag)
    m_horaria = m_original['(-0.125, 38.625) ' + tag].loc[index1]
    print(i)
    mae_prueba(m_horaria, m_interpolada_horaria)
    if i == 4:
       plot_diagonal(m_horaria, m_interpolada_horaria, 'interpolacion_cubica_diagonal_modulo')
       plot_comparacion(m_horaria, m_interpolada_horaria, 'interpolacion_cubica_comparacion_modulo') 
    else:
        plot_diagonal(m_horaria, m_interpolada_horaria, 'interpolacion_cubica_diagonal_' + tag)
        plot_comparacion(m_horaria, m_interpolada_horaria, 'interpolacion_cubica_comparacion_' + tag)
    


   