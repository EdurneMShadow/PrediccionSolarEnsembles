#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import datetime
import pandas as pd
import desagregar as lib

from pvlib.location import Location


control = pd.read_csv('/gaa/home/edcastil/datos/20131231.mdata.det_trihorario.csv', index_col=0)
control_cols = list(control.columns)
coordenadas = [i[:-4] for i in control_cols]
coordenadas = [i[:-1] for i in coordenadas if i[-1]== ' ']
coordenadas = list(set(coordenadas))
grid=[]
for i in coordenadas:
    tupla = eval(i)
    grid.append([tupla[1], tupla[0]])


geopot = pd.read_csv('/gaa/home/edcastil/datos/geopot.csv', sep=' ', index_col=0)
cs_total = pd.DataFrame()
cols=[]
for [lat, lon, geo] in geopot.loc[:, ['Latitude', 'Longitude', 'Geopotential']].values:
    if [lat, lon] in grid:
        tus = Location(lat, lon, 'UTC', geo/9.8, 'Spain')
        times = pd.DatetimeIndex(start='2015-01-01', end='2016-01-01', freq='H', tz=tus.tz)
        cs = tus.get_clearsky(times) 
        cols.append('('+ str(lon) + ', ' + str(lat) + ') CS_H')
        cs_total = pd.concat([cs_total, pd.DataFrame(cs['ghi'].values)], axis=1)

inicio = datetime.datetime(2015,1,1)
fin = datetime.datetime(2016,1,1)
index = lib.crear_indice_anio(inicio, fin, tipo='h')
cs_total = cs_total.loc[cs_total.index[:-1]]

cs_total.index = index
cs_total.columns = cols
cs_total.to_csv('/gaa/home/edcastil/datos/cs_2015_nuevo.csv')