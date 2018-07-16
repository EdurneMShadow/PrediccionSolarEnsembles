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
#name = '/gaa/home/edcastil/datos/ensembles_nuevos/20151231.mdata.ens_desacc_'
name = '/gaa/home/edcastil/datos/ensembles_nuevos/20151231.mdata.ens_desacc_resolucion_'
df = pd.read_csv(name + num_ens + '.csv', index_col=0)
df_control = pd.read_csv('/gaa/home/edcastil/datos/control/20131231.mdata.control_desagregado_resolucion.csv', index_col=0)

##sacar sus columnas
#cols = df.columns
#cols_control = df_control.columns
##cambiar el formato del nombre de las columas
#for i in cols:
#    a = i[:-1].upper()
#    pos = a.find('(')
#    var = a[:pos]
#    coor = eval(a[pos:])
#    coor = str((coor[1],coor[0]))
#    new_i = coor + ' ' + var
#    df = df.rename(columns={i:new_i})
#
#l_delete = []
#for i in df.columns:
#    if i not in cols_control:
#        l_delete.append(i)
#        
##cargar lista de coordenadas a borrar
##f = open('/gaa/home/edcastil/scripts/coordenadas_no_control.pkl', 'rb')
##f = open('coordenadas_no_control.pkl', 'rb')    
##l_coor = pickle.load(f)
#
##tags = ['FDIR', 'CDIR', 'TCC', 'U10', 'V10', 'T2M', 'SSRD', 'SSR']
#
##l_delete = []
##for i in l_coor:
##    for j in tags:
##        l_delete.append(str(i) + ' ' + j)
#
#df = df.drop(l_delete, axis=1)
#df.to_csv(name + 'resolucion_' + num_ens + '.csv')


#reordenar coordenadas
df_new = pd.DataFrame(index = df.index, columns = df_control.columns)
for i in df_control.columns:
    df_new[i] = df[i]
    
df_new.to_csv(name + num_ens + '.csv')











