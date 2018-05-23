#!/usr/bin/env python3
#encoding: utf-8
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import glob
import pandas as pd
import numpy as np
import cambio_resolucion as cr
import datetime
import desagregar_ensembles as des
import desagregar as lib

assert (len(sys.argv) >= 2), 'Debe haber un argumento'
name = sys.argv[1]

tags_ens = ['fdir', 'cdir', 'tcc', 'u10', 'v10', 't2m', 'ssrd', 'ssr']
inicio = datetime.datetime(2015,1,1)
fin = datetime.datetime(2015,12,31) 
nombre_parcial = '/gaa/home/edcastil/datos/ensembles_nuevos/20151231.mdata.ens_'

#for name in glob.glob("/gaa/home/alecat/edurne_ens/dfs/*"):
df = pd.read_csv(name, index_col=0)
#df_resolucion = cr.cambio_resolucion(df, tags_ens, "NOMBRE", inicio, fin, step=3)

columnas = []
for i in df.columns:
    for j in tags_ens:
        if i.find(j) is not -1:
            ens_number = i[-1:]
            no_number = i[:-1]
            pos = no_number.find('(')
            coor = eval(no_number[pos:])
            lon = coor[0]
            lat = coor[1]
            lon = abs(lon) - abs(int(lon))
            lat = abs(lat) - abs(int(lat))
            if (lon == 0.5 or lon == 0.0) and (lat == 0.5 or lat == 0.0):
                columnas.append(i)
                break
df_resolucion = df[columnas]
index = lib.obtener_indice_luminoso(inicio, fin, step=3)
df_resolucion.index = index

nombre = nombre_parcial + 'resolucion' + '_' + ens_number + '.csv'
df_resolucion.to_csv(nombre)

nuevo_nombre= nombre_parcial + 'desacc' + '_' + ens_number + '.csv'
df_desagregado = des.desagregar(nombre, nuevo_nombre)