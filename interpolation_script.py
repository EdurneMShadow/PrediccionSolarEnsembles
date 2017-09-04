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

def main(var_1, var_2):
    """Converts hourly variable var_1 to 3-hour accumulated and
    interpolates this with the hourly values in var_2
    """
    df = pd.read_csv("22Marzo_interpolacion.csv", sep=';')
    
    var_1 = df['(-0.125. 38.625) ' + var_1]
    var_2 = df['(-0.125. 38.625) ' + var_2]
    
    var_1_acc_3h = var_2_var_acc_3h(var_1)
    
    var_1_int = interpolar(var_1_acc_3h, var_2)
    
    check(var_1, var_1_int)
    
    return var_1_int


def var_2_var_acc_3h_prev_2(var):
    """Accumulates 3--hour values of var generating 8 values
    with accumulations at hours 0, 3, 6, 9, 12, 15, 18 and 21.
    Only works for the 0-UTC time zone
    """
    hours_0 = [var[i] for i in range(4)] + [var[i] for i in range(21,24)]   
    
    var_acc_3h = []
    
    for h in range(8):
        var_acc_3h.append( var[3*h-2 : 3*h+1].sum() )
        
    return np.array(var_acc_3h)
              

def var_2_var_acc_3h(var):
    """Accumulates 3--hour values of var generating 8 values
    with accumulations at hours 0, 3, 6, 9, 12, 15, 18 and 21.
    Only works for the 0-UTC time zone.
    Simplemente suma de tres en tres. Por ejemplo, en el 3, la suma es de 1,2,3
    """
    try:
        assert len(var) == 24
        #comprueba que las cuatro primeras horas del dia y las tres últimas sean cero.
        hours_0 = [var[i] for i in range(4)] + [var[i] for i in range(21,24)]
        assert sum(abs(np.array(hours_0))) == 0
        
        #está eliminando el primer valor y poniéndolo al final. La lista ahora va de 1 a 24.
        var_aux = np.array(list(var[1 : ]) + [0.])
        var_acc_3h = []
    
        #recorre por las horas trihorarias
        for h in range(8):
            #var_acc_3h.append( var_aux[3*h-2 : 3*h+1].sum() )
            var_acc_3h.append( var_aux[3*h : 3*h+3].sum() )
        #elimina el último elemento y añade un cero al inicio de la lista
        var_acc_3h = [0.] + var_acc_3h[ : -1]
        return np.array(var_acc_3h)
    
    except AssertionError:
        print("argument must have 24 values and values 0-3 and ",
              "21-23 of argument must be 0")

              
def var_2_var_acc_3h_prev(var):
    """Accumulates 3--hour values of var generating 8 values
    with accumulations at hours 0, 3, 6, 9, 12, 15, 18 and 21.
    Only works for the 0-UTC time zone
    """
    try:
        assert len(var) == 24
        
        hours_0 = [var[i] for i in range(4)] + [var[i] for i in range(21,24)]
        #print(hours_0, sum(abs(hours_0)))
        
        assert sum(abs(np.array(hours_0))) == 0
        
        var_acc_3h = []
    
        for h in range(8):
            var_acc_3h.append( var[3*h-2 : 3*h+1].sum() )
        
        return np.array(var_acc_3h)
    
    except AssertionError:
        print("argument must have 24 values and values 0-3 and ",
              "21-23 of argument must be 0")
              
    
def var_acc_3h_2_var_acc_h(var_acc_3h):
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
        l_var_acc_3h = list(var_acc_3h[1 : ]) + [0.] 
        
        var_acc_h = []
        for h in range(24):
            var_acc_h.append(l_var_acc_3h[h//3])
            
        #add first 0. and drop last 0.
        return np.array([0.] + var_acc_h[ : 23])
    
    except AssertionError:
        print("argument must have 8 values")

        
def var_acc_3h_2_var_acc_h_prev(var_acc_3h):
    """Giver  hourly values of a variable with 8 values accumulated every three hours.
    The values at hours h-2, h-1, h for 0, 3, 6, 9, 12, 15, 18 and 21 will be 
    the same.
    It is used in order to apply a CS interpolation of the form 
    var_acc_int = var_acc_h * cs_coef
    """
    var_acc_h = []
    for h in range(22):
        var_acc_h.append(var_acc_3h[(h+2)//3])
    var_acc_h.append(0)
    var_acc_h.append(0)
    
    return np.array(var_acc_h)
    
    
def denom_cs_interp(cs):
    """Computes the denominator cs_acc in order to apply a CS interpolation of the form 
    var_acc_int = var_acc * cs / cs_acc.
    
    cs can be any variable although 'CS H' is expected.
    """
    #acumula de tres en tres
    cs_acc_3h = var_2_var_acc_3h(cs)
    #lo estira para que sean 24h.
    cs_acc_h  = var_acc_3h_2_var_acc_h(cs_acc_3h)

    #cambia los ceros por unos para que no haya problemas en la división.
    cs_acc_h[ cs_acc_h == 0. ] = 1.

    return cs_acc_h

    
def interpolar(var_1_acc_3h, var_2_h):
    """Interpolates 3-hour accumulated values in var_1_3h w.r.
    the hourly values in var_2_h.
    
    For instance var_1_acc could be 8 accumulated SSRD and var_2_h 24
    hourly values.
    """
    try:
        #comprueba que la primera variable sea trihoraria y la segunda sea horaria.
        assert len(var_1_acc_3h) == 8 and len(var_2_h) == 24
        var_1_acc_h = var_acc_3h_2_var_acc_h(var_1_acc_3h)
        var_2_acc_h = denom_cs_interp(var_2_h)
        
        return var_1_acc_h * var_2_h / var_2_acc_h
    
    except AssertionError:
        print("args must have 8 and 24 values respectively")
    
def check(var, var_int):
    print("MAE: ", abs(var - var_int).mean())
    print("RMAE: ", abs(var - var_int).mean()/var.mean())
    
    plt.title("real (r) vs interp (b) values")
    plt.plot(var, 'r', var_int, 'b')
    plt.show()

    
#################################################### cubic splines
#definir interpolador para las 3-horas de 0 a 21
#hours_3h = [0, 3, 6, 9, 12, 15, 18, 21]
#horas de 0 a 21
#hours = [i for i in range(22)]
    
def cubic_spline_interpol(hours, hours_3h, var):
    cubic_interp = CubicSpline(hours_3h, var[hours_3h])
    return cubic_interp(hours) 

    
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
        print(var_1, var_2)
        main(var_1, var_2)
    else:
        print("args: nombre_1 nombre_2")