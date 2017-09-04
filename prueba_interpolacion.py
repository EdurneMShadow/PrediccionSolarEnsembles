# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:22:05 2017

@author: Edurne
"""
import desagregar as lib
import pandas as pd
import numpy as np
import DataMatrix_NWP as dm

def main():
    cs = pd.read_csv('./Prueba_interpolacion/matriz_punto_cs.csv', index_col = 0)
    cs3 = pd.read_csv('./Prueba_interpolacion/matriz_punto_cs3.csv', index_col = 0)
    m_original = pd.read_csv('./Prueba_interpolacion/matriz_punto_original.csv', index_col = 0)
    m_trihoraria = pd.read_csv('./Prueba_interpolacion/matriz_punto_trihoraria.csv', index_col = 0)

    m_horaria_interpolada = interpolacion(cs, cs3,m_trihoraria)
    mae_prueba(m_original, m_horaria_interpolada)


def calcular_medias_radiaciones(matrix, tags):
    '''PENDIENTE DE REVISAR'''
    medias = list(matrix.mean())
    media_radiaciones = {}
    for i, tag in enumerate(tags):
        media_radiaciones[tag] = medias[i]
    return media_radiaciones

'''Medias de las radiaciones trihorarias'''
#m_fdir = pd.DataFrame(m_original.mean()[0], index=index1, columns=fdir.columns)
#m_fdir.to_csv('m_fdir.csv')

def interpolacion(cs, cs_3h, r_3h):
    #Preparación de dataframes para que tengan las mismas dimensiones
    #a = pd.concat([fdir,fdir,fdir])
    #a.sort_index()
    cs_3h = pd.concat([cs_3h, cs_3h, cs_3h])
    cs_3h = cs_3h.sort_index()
    cs_3h = pd.concat([cs_3h[2:], cs_3h[:2]]) #pone las dos primeras filas al final

    dates = pd.date_range('20150101','20160101',freq='1H')[:-1]
    index1 = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])
    cs_3h.index = index1

    cs = cs[17520:]

    division = cs/cs_3h
    division_extendida = pd.concat([division,division,division,division,division], axis=1)
    division_extendida.columns = r_3h.columns

    r_3h = pd.concat([r_3h, r_3h, r_3h])
    r_3h = r_3h.sort_index()
    r_3h = pd.concat([r_3h[2:], r_3h[:2]]) #pone las dos primeras filas al final

    r_3h.index = index1

    interpolado = division*r_3h
    return interpolado

'''MAE'''
def mae_prueba(original, interpolado):
    print("MAE: ", abs(original - interpolado).mean())
    print("RMAE: ", abs(original - interpolado).mean()/original.mean())


'''MAE diario'''
def mae_diario_prueba(original, interpolado):
#r_original = pd.read_csv('./Plots/radiacion_original.csv', index_col=0)
#r_interpolado = pd.read_csv('./Plots/radiacion_interpolada.csv', index_col=0)
#tags = ['FDIR','CDIR','SSR','SSRC','SSRD']
    mae_dias = {}
    for i in range(5):
        mae = lib.MAE_diario(interpolado[interpolado.columns[i]], original[original.columns[i]])
        mae_dias[tags[i]] = mae


'''Obtener datos para el 22 de Marzo
r_original = pd.read_csv('./Plots/radiacion_original.csv', index_col=0)
indice  = lib.obtener_dia_completo(2015032200)
dia = r_original.loc[indice]
dia.to_csv('22Marzo_radiacion.csv', sep=';')

cs = pd.read_csv('./Plots/cs_horario.csv', index_col=0)
dia_cs = cs.loc[indice]
dia_cs.to_csv('22Marzo_cs.csv', sep=';')


cs3 = pd.read_csv('./Plots/cs_trihorario.csv', index_col=0)
index = [2015032200,2015032203,2015032206,2015032209,2015032212,2015032215,2015032218,2015032221]
dia_cs3 = cs3.loc[index]
dia_cs3.to_csv('22Marzo_cs3.csv', sep=';')


r_interpolado = pd.read_csv('./Plots/radiacion_interpolada.csv', index_col=0)
dia_interpolado = r_interpolado.loc[indice]
dia_interpolado.to_csv('22Marzo_interpolado.csv', sep=';')


r_trihorario = pd.read_csv('./Plots/radiacion_trihoraria.csv', index_col=0)
dia_trihorario = r_trihorario.loc[index]
dia_trihorario.to_csv('22Marzo_trihorario.csv', sep=';')
'''
if __name__ == '__main__':
    main()
