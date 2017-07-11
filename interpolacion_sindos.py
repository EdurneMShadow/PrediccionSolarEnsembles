#!/usr/bin/env python3
# encoding: utf-8

import DataMatrix_NWP as dm
import datetime
import pandas as pd
import numpy as np
import sunrise as sr
import libdata as ut
import desagregar as lib

#Carga de matrices
df = pd.DataFrame()
cs_1h = ut.load_CS(df, shft=0)
cs_1h = cs_1h[:-1]
cs_3h = pd.read_csv('cs_3h_acc.csv', index_col=0)
latlon = dm.select_pen_grid()
matrix=dm.DataMatrix(datetime.datetime(2015,12,31),'/gaa/home/edcastil/','/gaa/home/edcastil/',ifexists=True,model='deterministic',suffix='.det_3h_acc', tags = ['FDIR', 'CDIR','SSRD', 'SSR', 'SSRC'], latlons = latlon)
matrix = matrix.dataMatrix

#interpolacion
lib.interpolacion(cs_1h, cs_3h, matrix)
