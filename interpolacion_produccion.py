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
    prod_interpolada.to_csv('Produccion/prod_interpolada_trihorario.csv')
    
    check(prod_h, prod_interpolada)
    
    return prod_interpolada

def extender_trihorario_a_horario(matrix_3h):
    new_matrix = []
    for i in range (0,matrix_3h.shape[0],6):
        index = matrix_3h.index[i:i+6]
        submatrix = matrix_3h.loc[index]
        new_index = np.arange(6)
        submatrix.index = new_index
        new_matrix.append(submatrix.loc[0].values[0])
        submatrix = submatrix.loc[1:]
              
        for h in range(1,6):
            for i in range(3):
                new_matrix.append(submatrix.loc[h].values[0])
                
    inicio = datetime.datetime(2015,1,1)
    fin = datetime.datetime(2015,12,31)
    index_df = lib.obtener_indice_luminoso(inicio, fin, step=1)
    df = pd.DataFrame(new_matrix, index = index_df, columns = matrix_3h.columns)
    return df


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

#    y = np.arange(0,24,1)
#    plt.figure()
#    plt.title('Producción predicha vs real 2015 (interpolacion)')
#    plt.xlabel('Horas')
#    plt.ylabel('Producción')
#    _ = plt.plot(y,var_original, label = 'original')
#    _ = plt.plot(y, var_interpolada, label = 'interpolado')
#    plt.legend(loc = 'best')
#    nombre = 'Imagenes/'+ 'determinista_trihorario_interpolado.pdf'
#    plt.savefig(nombre)    
#################################################### main
if __name__ == '__main__':
    prod_3h = pd.read_csv('Produccion/Prod_2015_trihorario.csv', index_col=0)
    prod_h = pd.read_csv('Produccion/Prod_2015_resolucion.csv', index_col=0)
    cs = pd.read_csv('cs_2015_mean.csv', index_col=0)
    cs_acc = pd.read_csv('cs_acc_2015_mean.csv', index_col=0)
    main(prod_3h, prod_h, cs, cs_acc)