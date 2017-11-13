# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:31:51 2017

@author: Edurne
"""
import numpy as np
import pandas as pd
import desagregar as lib


def acumular_trihorario(matrix, indice, nombre):
    '''Dada una matrix determinista horaria y un índice de fechas de un año trihorario, se devuelve 
    y se guarda una matriz determinista trihoraria. El nombre tiene que ser un string acabado en .csv'''
    new_matrix = []
    for i in range (0,matrix.shape[0],16):
        index_m = matrix.index[i:i+16]
        submatrix = matrix.loc[index_m]       
        new_matrix.append(submatrix.loc[index_m[0]])
        submatrix = submatrix.loc[index_m[1:]]
        index_luminoso = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        submatrix.index = index_luminoso
        for h in range(5):
            new_matrix.append(submatrix[3*h : 3*h+3].sum())
    data = pd.DataFrame(new_matrix, index=indice)
    data.to_csv('/gaa/home/edcastil/datos/' + nombre)
    return data
            