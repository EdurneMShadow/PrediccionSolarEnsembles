#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import desagregar as lib
import datetime


def main(y_pred, cs, cs_acc):
    y_pred = convertir_a_df(y_pred)
    y_pred[y_pred['y_pred'] < 0] = 0.0
    y_pred_h = extender_trihorario_a_horario(y_pred)
    cs_acc[cs_acc['CS_H_mean'] == 0] = 1.0
    cs_acc_h = extender_trihorario_a_horario(cs_acc)
    
    y_interpolada = interpolar(cs, cs_acc_h, y_pred_h)
    y_interpolada.to_csv( 'y_interpolada_control.csv')
    check(y_test, y_interpolada)


def convertir_a_df(y_array):
    inicio = datetime.datetime(2015,1,1)
    fin = datetime.datetime(2015,12,31)
    index = lib.obtener_indice_luminoso(inicio, fin, step=3)
    df = pd.DataFrame(y_array, index = index, columns = ['y_pred'])
    return df

def extender_trihorario_a_horario(matrix_3h):
    '''Dada una matriz trihoraria de 6 a 21, devuelve una matriz horaria (de 6 a 21) calculada de la siguiente forma:
        - 6 se queda igual
        - 7 y 8 van a tener el mismo valor que 9
        mode=0 -> y_pred
        mode=1 -> cs_acc'''

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
    #Cargar datos
    cs = pd.read_csv('/gaa/home/edcastil/datos/cs_2015_mean.csv', index_col=0)
    cs_acc = pd.read_csv('/gaa/home/edcastil/datos/cs_acc_2015_mean.csv', index_col=0)
    f = open('/gaa/home/edcastil/scripts/comparaciones_svr_test_control.pkl', 'rb')
    y_pred = pickle.load(f)
    y_pred = y_pred[1]
    y_test = pd.read_csv('/gaa/home/edcastil/datos/Prod_2015_resolucion.csv', index_col=0)
    main(y_pred, cs, cs_acc)