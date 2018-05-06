#!/usr/bin/env python3
# encoding: utf-8
"""
Created on Sun May  6 20:15:35 2018

@author: Edurne
"""
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import pandas as pd
import pickle

num_ens = sys.argv[1]
#cargar csv a tratar
name = '/gaa/home/edcastil/datos/ensembles/20151231.mdata.ens_desacc_'
df = pd.read_csv(name + num_ens + 'csv', index_col=0)

#sacar sus columnas

cols = df.columns
for i in cols:
    a = i[:-2].upper()
    pos = a.find('(')
    var = a[:pos]
    coor = eval(a[pos:])
    coor = str((coor[1],coor[0]))
    new_i = coor + ' ' + var
    df.rename(columns={i:new_i})
    
#cargar lista de coordenadas a borrar
f = open('/gaa/home/edcastil/scripts/coordenadas_no_control.pkl', 'rb')
#f = open('coordenadas_no_control.pkl', 'rb')    
l_coor = pickle.load(f)

tags = ['FDIR', 'CDIR', 'TCC', 'U10', 'V10', 'T2M', 'SSRD', 'SSR']

l_delete = []
for i in l_coor:
    for j in tags:
        l_delete.append(str(i) + ' ' + j)
        
df.drop(l_delete, axis=1)
df.to_csv(name + '_resolucion_' + num_ens + '.csv')