# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:59:11 2017

@author: Edurne
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

def main(prod, cs, cs_acc):
   
    prod_3h = acumular_trihorario(prod)
    
    prod_interpolada = interpolar(prod_3h, cs, cs_acc)
    
    check(prod, prod_interpolada)
    
    return prod_interpolada

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

def acumular_trihorario(df):
    """Accumulates 3--hour values of var generating 8 values
    with accumulations at hours 0, 3, 6, 9, 12, 15, 18 and 21.
    Only works for the 0-UTC time zone.
    Simplemente suma de tres en tres. Por ejemplo, en el 3, la suma es de 1,2,3
    """
    df_horario = list(df['Prod'])
    
    #está eliminando el primer valor y poniéndolo al final. La lista ahora va de 1 a 24.
    var_aux = np.array(variable_horaria[1 : ] + [0.])
    var_acc_3h = []

    #recorre por las horas trihorarias
    for h in range(8):
        var_acc_3h.append(var_aux[3*h : 3*h+3].sum())
        
    #elimina el último elemento y añade un cero al inicio de la lista
    var_acc_3h = [0.] + var_acc_3h[ : -1]
    return np.array(var_acc_3h)     
    
def extender_trihorario_a_horario(var_acc_3h):
    """Giver  hourly values of a variable with 8 values accumulated every three hours.
    The values at hours h-2, h-1, h for 0, 3, 6, 9, 12, 15, 18 and 21 will be 
    the same.
    It is used in order to apply a CS interpolation of the form 
    var_acc_int = var_acc_h * cs_coef
    Es decir, 1 y 2 van a tener el mismo valor que 3. Es la forma de extender a 24h.
    """
    try:
        assert len(var_acc_3h) == 8, "argument must have 8 values"
        
        #drop first 0 and add last 0. to work with 8 consecutive 3-h groups
        var_acc_3h_aux = list(var_acc_3h[1 : ]) + [0.] 
        
        var_acc_h = []
        for h in range(24):
            var_acc_h.append(var_acc_3h_aux[h//3])
            
        #add first 0. and drop last 0.
        return np.array([0.] + var_acc_h[ : 23])
    
    except AssertionError:
        print("argument must have 8 values")

                
def calcular_cs_trihorario_expandido(cs):
    """Computes the denominator cs_acc in order to apply a CS interpolation of the form 
    var_acc_int = var_acc * cs / cs_acc.
    
    cs can be any variable although 'CS H' is expected.
    """
    #acumula de tres en tres
    cs_acc_3h = acumular_trihorario(cs)
    #lo estira para que sean 24h.
    cs_acc_h  = extender_trihorario_a_horario(cs_acc_3h)

    #cambia los ceros por unos para que no haya problemas en la división.
    cs_acc_h[ cs_acc_h == 0. ] = 1.

    return cs_acc_h

    
def interpolar(var_radiacion_acc_3h, var_cs):
    """Interpolates 3-hour accumulated values in var_1_3h w.r.
    the hourly values in var_2_h.
    
    For instance var_1_acc could be 8 accumulated SSRD and var_2_h 24
    hourly values.
    """
    try:
        #comprueba que la primera variable sea trihoraria y la segunda sea horaria.
        assert len(var_radiacion_acc_3h) == 8 and len(var_cs) == 24
        var_radiacion_acc_h = extender_trihorario_a_horario(var_radiacion_acc_3h)
        var_cs_acc_h = calcular_cs_trihorario_expandido(var_cs)
        
        return var_radiacion_acc_h * var_cs / var_cs_acc_h
    
    except AssertionError:
        print("args must have 8 and 24 values respectively")
    
def check(var_original, var_interpolada, nombre_variable_radiacion):
    print("MAE: ", abs(var_original - var_interpolada).mean())
    print("RMAE: ", abs(var_original - var_interpolada).mean()/var_original.mean())
    
    y = np.arange(0,24,1)
    plt.figure()
    plt.title(nombre_variable_radiacion)
    plt.xlabel('Horas')
    plt.ylabel('Radiación')
    _ = plt.plot(y,var_original, label = 'original')
    _ = plt.plot(y, var_interpolada, label = 'interpolado')
    plt.legend(loc = 'best')
    nombre = 'Imagenes/'+ nombre_variable_radiacion + '_interpolado.pdf'
    plt.savefig(nombre)

    
#################################################### main
if __name__ == '__main__':
    prod = pd.read_csv('Produccion/Prod_2015_resolucion.csv', index_col=0)
    cs = pd.read_csv('cs_2015_mean.csv', index_col=0)
    cs_acc = pd.read_csv('cs_acc_2015_mean.csv', index_col=0)
    main(prod, cs, cs_acc)