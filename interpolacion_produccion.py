#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import desagregar as lib

from scipy.interpolate import CubicSpline

def main(prod_3h, prod_h, cs, cs_acc):
   
    prod_3h_extendida = extender_trihorario_a_horario(prod_3h)
    cs_acc[cs_acc['CS_H_mean'] == 0] = 1.0
    cs_acc_h = extender_trihorario_a_horario(cs_acc)
    prod_interpolada = interpolar(cs, cs_acc_h, prod_3h_extendida)
    
    check(prod_h, prod_interpolada)
    
    return prod_interpolada

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


def interpolar(cs, cs_acc_h, y_pred_h):
    try:
        #comprueba que la primera variable sea trihoraria y la segunda sea horaria.
        assert len(cs) == len(cs_acc_h) == len(y_pred_h)
        
        interpolada = y_pred_h.values * cs.values / cs_acc_h.values
        inicio = datetime.datetime(2015,1,1)
        fin = datetime.datetime(2015,12,31)
        index_df = lib.obtener_indice_luminoso(inicio, fin, step=1)
        df = pd.DataFrame(interpolada, index = index_df, columns = ['Produccion_interpolada'])
        return df
    except AssertionError:
        print("Los argumentos deben tener el mismo tamano")
    
def check(original, interpolada):
    print("MAE: ", abs(original.values - interpolada.values).mean())
    print("RMAE: ", abs(original.values - interpolada.values).mean()/original.values.mean())

    
#################################################### main
if __name__ == '__main__':
    prod_3h = pd.read_csv('Produccion/Prod_2015_trihoraria.csv', index_col=0)
    prod_h = pd.read_csv('Produccion/Prod_2015_resolucion.csv', index_col=0)
    cs = pd.read_csv('cs_2015_mean.csv', index_col=0)
    cs_acc = pd.read_csv('cs_acc_2015_mean.csv', index_col=0)
    main(prod_3h, prod_h, cs, cs_acc)