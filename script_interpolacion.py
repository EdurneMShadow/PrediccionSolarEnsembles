#!/usr/bin/env python3
# encoding: utf-8

import desagregar as lib
import DataMatrix_NWP as dm
import datetime
import pandas as pd
import numpy as np
import sunrise as sr
import libdata as ut

#Carga de matrices
df = pd.DataFrame()
cs_1h = ut.load_CS(df, shft=0)
cs_3h = pd.read_csv('cs_trihorario_acumulado_2015.csv', sep=',')
dates = pd.date_range('20150101','20160101',freq='3H')[:-1]
indice = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])
cs_3h.index = indice
latlon = dm.select_pen_grid()
matrix=dm.DataMatrix(datetime.datetime(2015,12,31),'/gaa/home/edcastil/','/gaa/home/edcastil/',ifexists=True,model='deterministic',suffix='.det_3h_acc', tags = ['FDIR', 'CDIR','SSRD', 'SSR', 'SSRC'], latlons = latlon)

#interpolación
interpolacion(cs, cs_3h, r_3h)
