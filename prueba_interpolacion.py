# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:22:05 2017

@author: Edurne
"""
import desagregar as lib
import pandas as pd
import numpy as np
import DataMatrix_NWP as dm


dates = pd.date_range('20150101','20160101',freq='3H')[:-1]
index3 = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])

dates = pd.date_range('20150101','20160101',freq='1H')[:-1]
index1 = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])

dates = pd.date_range('20130101','20160101',freq='1H')[:-1]
indexcs = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])

'''Carga del clear-sky horario'''
cs = pd.read_csv('./Prueba_interpolacion/columna_cs1.csv', header=None, index_col = 0)
cs.columns = ['(-0.125, 38.625) CS H']

'''Carga del clear-sky trihorario'''
cs3 = pd.read_csv('./Prueba_interpolacion/columna_cs3.csv', header=None, index_col=0)
cs3.index = index3
cs3.columns = ['(-0.125, 38.625) CS H']

'''Carga de las radiaciones trihorarias'''
fdir = pd.read_csv('./Prueba_interpolacion/fdir.csv', index_col=0)
cdir = pd.read_csv('./Prueba_interpolacion/cdir.csv', index_col=0)
ssr = pd.read_csv('./Prueba_interpolacion/ssr.csv', index_col=0)
ssrc = pd.read_csv('./Prueba_interpolacion/ssrc.csv', index_col=0)
ssrd = pd.read_csv('./Prueba_interpolacion/ssrd.csv', index_col=0)

'''Medias de las radiaciones trihorarias'''
m_fdir = pd.DataFrame(fdir.mean()[0], index=index1, columns=fdir.columns)
m_fdir.to_csv('m_fdir.csv')

m_cdir = pd.DataFrame(cdir.mean()[0], index=index1, columns=cdir.columns)
m_cdir.to_csv('m_cdir.csv')

m_ssr = pd.DataFrame(ssr.mean()[0], index=index1, columns=ssr.columns)
m_ssr.to_csv('m_ssr.csv')

m_ssrc = pd.DataFrame(ssrc.mean()[0], index=index1, columns=ssrc.columns)
m_ssrc.to_csv('m_ssrc.csv')

m_ssrd = pd.DataFrame(ssrd.mean()[0], index=index1, columns=ssrd.columns)
m_ssrd.to_csv('m_ssrd.csv')

def interpolacion(cs, cs_3h, r_3h):
    dates = pd.date_range('20150101','20160101',freq='1H')[:-1]
    index = np.array([int(d.strftime("%Y%m%d%H")) for d in dates]) #índice horario
    col = []
    print('df creado!')
    var = 0
    for i in index:
        cs1 = cs.loc[i][0]
        
        terminacion = str(i)[8:]
        if int(terminacion) > var:
            if var == 21:
                var = 0
            else:
                var+=3
        if len(str(var)) < 2:
            var_string = '0' + str(var)
        else:
            var_string = str(var)
        indice = int(str(i)[:8] + var_string)
        
        cs3 = cs_3h.loc[indice][0]
        r = r_3h.loc[indice][0]
        if cs3 == 0.0: 
            col.append(0.0)
        else:
            print('Resultado: ' + str(cs1/cs3*r))
            col.append(cs1/cs3*r)
    df_interpolado = pd.DataFrame(col, index = index, columns = ['(-0.125, 38.625) FDIR'])
    return df_interpolado
  
'''MAE'''      
dates = pd.date_range('20150101','20160101',freq='1H')[:-1]
index = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])

fdir_interpolado = pd.read_csv('./Prueba_interpolacion/fdir_interpolado.csv', index_col=0)
var = matrix_original['(-0.125, 38.625) FDIR']
original = pd.DataFrame(var.values,index=index, columns=[var.name])
lib.MAE(fdir_interpolado, original)
        
m_fdir = pd.read_csv('./Prueba_interpolacion/m_fdir.csv', index_col=0)       
lib.MAE(m_fdir, original)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
