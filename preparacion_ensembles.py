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

assert (len(sys.argv) >= 3), 'Debe haber dos argumento'
name = sys.argv[1]
ens_number = sys.argv[2]

tags_ens = ['fdir', 'cdir', 'tcc', 'u10', 'v10', 't2m', 'ssrd', 'ssr']
inicio = datetime.datetime(2015,1,1)
fin = datetime.datetime(2015,12,31) 
#nombre_parcial = '/gaa/home/edcastil/datos/ensembles_nuevos/20151231.mdata.ens_'
nombre_parcial = '/gaa/home/alecat/edurne_ens/dfs/2015123100.mdata.ens_1.csv' 
#for name in glob.glob("/gaa/home/alecat/edurne_ens/dfs/*"):
df = pd.read_csv(name, index_col=0)
df_control = pd.read_csv('/gaa/home/edcastil/datos/control/20131231.mdata.control_desagregado_resolucion.csv', index_col=0)
#df_resolucion = cr.cambio_resolucion(df, tags_ens, "NOMBRE", inicio, fin, step=3)

#columnas = []
#for i in df.columns:
#    for j in tags_ens:
#        if i.find(j) is not -1:
#            ens_number = i[-1:]
#            no_number = i[:-1]
#            pos = no_number.find('(')
#            coor = eval(no_number[pos:])
#            lon = coor[0]
#            lat = coor[1]
#            lon = abs(lon) - abs(int(lon))
#            lat = abs(lat) - abs(int(lat))
#            if (lon == 0.5 or lon == 0.0) and (lat == 0.5 or lat == 0.0):
#                columnas.append(i)
#                break
#df_resolucion = df[columnas]
#index = lib.obtener_indice_luminoso(inicio, fin, step=3)
#df_resolucion.index = index
#
#nombre = nombre_parcial + 'resolucion' + '_' + ens_number + '.csv'
#df_resolucion.to_csv(nombre)

#nuevo_nombre= nombre_parcial + 'desacc' + '_' + ens_number + '.csv'
nombre= nombre_parcial + 'desacc' + '_' + ens_number + '.csv'

des.desagregar(nombre_parcial, nombre) #desagregado

df_ens = pd.read_csv('/gaa/home/edcastil/datos/ensembles_nuevos/20151231.mdata.ens_desacc_1.csv', index_col=0)
ens_cols = df_ens.columns
control_cols = df_control.columns

def from_ens_2_control_var_names(s):
    pos_ini = s.find('(')
    pos_med = s.find(',')
    pos_fin = s.find(')')
    
    #return s[pos_ini : pos_fin+1] + ' ' + s[ : pos_ini].upper()
    return '(' + s[pos_med+2 : pos_fin] + ', ' + s[pos_ini+1 : pos_med] + ') ' + s[ : pos_ini].upper()

ens_cols_2 = []

for var in ens_cols:
    ens_cols_2.append( from_ens_2_control_var_names(var) )

df_ens_2 = pd.DataFrame(data=df_ens.values, index=df_ens.index, columns=ens_cols_2)

cols_both = list( set(control_cols).intersection(set(ens_cols_2)))

df_control_def = df_control[ sorted(cols_both)]
df_control_def.to_csv('prueba_control.csv')
df_ens_def = df_ens_2[ sorted(cols_both) ]
df_ens_def.to_csv('prueba_ens_1.csv')
