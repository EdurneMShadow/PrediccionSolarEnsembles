# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 13:43:52 2017

@author: Edurne
"""

import DataMatrix_NWP as dm
import datetime
import pandas as pd
import numpy as np
import time as t
import desagregar as lib


#tags_ens = ['FDIR', 'CDIR', 'TCC', 'U10', 'V10', 'T2M', 'SSRD', 'SSR']
#tags_det = ['FDIR', 'CDIR', 'TCC', 'U10', 'V10', 'T2M', 'SSRD', 'SSR', 'SSRC', 'v10']
#
#matrix_det_2013 = (dm.DataMatrix(datetime.datetime(2013,12,31), '/gaa/home/data/solar_ecmwf/', '/gaa/home/data/solar_ecmwf/', ifexists = True, model='deterministic', suffix='.det_noacc_vmodule'))

#nombre = '/gaa/home/edcastil/datos/20151231.mdata.det_resolucion.csv'
#inicio = datetime.datetime(2015,1,1)
#fin = datetime.datetime(2015,12,31)

def cambio_resolucion(matrix, tags, nombre, inicio, fin, step):
    columnas = []
    for i in matrix.columns:
        for j in tags:
            if i.find(j) is not -1:
                pos = i.find(j)
                coor = eval(i[:pos])
                lon = coor[0]
                lat = coor[1]
                lon = abs(lon) - abs(int(lon))
                lat = abs(lat) - abs(int(lat))
                if (lon == 0.5 or lon == 0.0) and (lat == 0.5 or lat == 0.0):
                    columnas.append(i)
                    break
    submatrix = matrix[columnas]
    index = lib.obtener_indice_luminoso(inicio,fin,step=step)
    
    submatrix = submatrix.loc[index]
    
    if step == 1:
        cols = np.load('/gaa/home/edcastil/datos/columnas_iguales.npy')
        submatrix = submatrix[cols]
    
    submatrix.to_csv(nombre)