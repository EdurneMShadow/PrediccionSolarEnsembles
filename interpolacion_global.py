#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


def main(y_pred, cs, cs_acc):
    y_pred_h = extender_trihorario_a_horario(y_pred)
    cs_acc_h = extender_trihorario_a_horario(cs_acc)
    
    y_interpolada = interpolar(cs, cs_acc_h, y_pred_h)
    nombre = 'y_interpolada_trihorario01.pkl'
    pickle.dump(y_interpolada, open(nombre, 'wb' ))
    check(y_test, y_interpolada)

def extender_trihorario_a_horario(matrix_3h):
    '''Dada una matriz trihoraria de 6 a 21, devuelve una matriz horaria (de 6 a 21) calculada de la siguiente forma:
        - 6 se queda igual
        - 7 y 8 van a tener el mismo valor que 9'''
    new_matrix = []
    for i in range (0,matrix_3h.shape[0],6):
        submatrix = matrix_3h[i:i+6]
        new_matrix.append(submatrix[0])
        submatrix = submatrix[1:]
              
        for h in range(0,len(submatrix)):
            for i in range(3):
                new_matrix.append(submatrix[h])
        
    return new_matrix

def interpolar(cs, cs_acc_h, y_pred_h):
    try:
        #comprueba que la primera variable sea trihoraria y la segunda sea horaria.
        assert len(cs) == len(cs_acc_h) == len(y_pred_h)
        
        return y_pred_h * cs / cs_acc_h
    
    except AssertionError:
        print("Los argumentos deben tener el mismo tamano")
        
def check(var_original, var_interpolada):
    print("MAE: ", abs(var_original - var_interpolada).mean())
    print("RMAE: ", abs(var_original - var_interpolada).mean()/var_original.mean())
    
    
#    y = np.arange(0,24,1)
#    plt.figure()
#    plt.title('Producción predicha vs real 2015 (interpolacion)')
#    plt.xlabel('Horas')
#    plt.ylabel('Producción')
#    _ = plt.plot(y,var_original, label = 'original')
#    _ = plt.plot(y, var_interpolada, label = 'interpolado')
#    plt.legend(loc = 'best')
#    nombre = 'Imagenes/'+ nombre_variable_radiacion + '_interpolado.pdf'
#    plt.savefig(nombre)

    
#################################################### main
if __name__ == '__main__':
    #Cargar datos
    cs = pd.read_csv('/gaa/home/edcastil/datos/cs_2015.csv', index_col=0)
    cs_acc = pd.read_csv('/gaa/home/edcastil/datos/cs_acc_2015.csv', index_col=0)
    f = open('/gaa/home/edcastil/scripts/comparaciones_svr_test_trihorario.pkl', 'rb')
    y_pred = pickle.load(f)
    y_pred = y_pred[1]
    y_test = pd.read_csv('/gaa/home/edcastil/datos/Prod_2015.csv', index_col=0)