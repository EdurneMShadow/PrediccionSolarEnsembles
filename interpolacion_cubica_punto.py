#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
%load_ext autoreload 
%autoreload 2
%matplotlib inline

%cd D:\Dropbox\catedra_myp\estancia_2016\edurne\interpol
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

def main(var_no_radiacion, fecha):
    """Convierte una variable horaria var_ a trihoraria acumulada 
    y la interpola con los valores horarios de var_cs
    """
    nombre_variable = var_no_radiacion
    df_original = pd.read_csv("Prueba_interpolacion/matriz_punto_original_viento.csv", index_col = 0)
    dia = obtener_dia_completo(fecha)
    df_original = df_original.loc[dia]   
    
    var_no_radiacion = df_original['(-0.125, 38.625) ' + var_no_radiacion]
    
    var_no_radiacion_acc_3h = acumular_trihorario(var_no_radiacion)
    
    lista_horas = np.arange(24)
    lista_horas_3 = np.arange(0,23,3)
    
    var_no_radiacion_interpolada = interpolacion_cubica(lista_horas, lista_horas_3, var_no_radiacion_acc_3h)
    
    check(var_no_radiacion, var_no_radiacion_interpolada, nombre_variable)
    
    return var_no_radiacion_interpolada

def obtener_dia_completo(fecha):
    '''Dada una fecha en formato YYYYMMDDHH entero, se devuelve una lista con todas las horas que conforman dicho día.
       Dicho de otra forma, se devuelve un índice para un día.
    '''
    dia = str(fecha)[:8]
    indice = []
    for i in range (24):
        if len(str(i)) < 2 :
            hora = '0' + str(i)
        else:
            hora = str(i)
        indice.append(int(dia + hora))
    return indice

def acumular_trihorario(variable_horaria):
    """Accumulates 3--hour values of var generating 8 values
    with accumulations at hours 0, 3, 6, 9, 12, 15, 18 and 21.
    Only works for the 0-UTC time zone.
    Simplemente suma de tres en tres. Por ejemplo, en el 3, la suma es de 1,2,3
    """
    try:
        assert len(variable_horaria) == 24
        variable_horaria = list(variable_horaria)
        
        #está eliminando el primer valor y poniéndolo al final. La lista ahora va de 1 a 24.
        var_aux = np.array(variable_horaria[1 : ] + [0.])
        var_acc_3h = []
    
        #recorre por las horas trihorarias
        for h in range(8):
            var_acc_3h.append(var_aux[3*h : 3*h+3].sum())
            
        #elimina el último elemento y añade un cero al inicio de la lista
        var_acc_3h = [0.] + var_acc_3h[ : -1]
        return np.array(var_acc_3h)
    
    except AssertionError:
        print("argument must have 24 values")
        
def interpolacion_cubica(horas, horas_3h, variable_3h):
    try:
        assert len(variable_3h) == 8
        interpolacion = CubicSpline(horas_3h, variable_3h)
        return interpolacion(horas)    
    except AssertionError:
        print('La variable debe estar en formato trihorario acumulado')              

    
def check(var_original, var_interpolada, nombre_variable):
    print("MAE: ", abs(var_original - var_interpolada).mean())
    print("RMAE: ", abs(var_original - var_interpolada).mean()/var_original.mean())
    y = np.arange(0,24,1)
    
    plt.figure()
    plt.title(nombre_variable)
    plt.xlabel('Horas')
    #plt.ylabel('Radiación')
    plt.xlim(0,23)
    _ = plt.plot(y,var_original, label = 'original')
    _ = plt.plot(y, var_interpolada, label = 'interpolado')
    plt.legend(loc = 'best')
    nombre = 'Imagenes/'+ nombre_variable + '_interpolado_cubico.pdf'
    plt.savefig(nombre)
    
   
#################################################### main
if __name__ == '__main__':
    if len(sys.argv) == 3:
        #print("leer dataframe y variables ...")
        #df = pd.read_csv("22Marzo_interpolacion.csv", sep=';')
        #
        #var_1 = df['(-0.125. 38.625) ' + sys.argv[1]]
        #var_2 = df['(-0.125. 38.625) ' + sys.argv[2]]
        #
        #print("imprimir vars...")
        print(var_no_radiacion, fecha)
        main(var_no_radiacion, fecha)
    else:
        print("args: nombre_1 fecha")