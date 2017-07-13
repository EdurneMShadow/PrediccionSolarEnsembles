#!/usr/bin/env python3
# encoding: utf-8

import desagregar as lib
import DataMatrix_NWP as dm
import datetime
import pandas as pd
import numpy as np
import sunrise as sr
import libdata as ut


latlon = dm.select_pen_grid()
matrix_interpolada = dm.DataMatrix(datetime.datetime(2015,12,31), '/gaa/home/edcastil/', '/gaa/home/edcastil/', ifexists = True, model='deterministic', suffix='.det_interpolado', tags = ['FDIR', 'CDIR','SSRD', 'SSR', 'SSRC'], latlons = latlon)
matrix_interpolada = matrix_interpolada.dataMatrix
matrix_original = dm.DataMatrix(datetime.datetime(2015,12,31), '/gaa/home/data/solar_ecmwf/', '/gaa/home/data/solar_ecmwf/', ifexists = True, model='deterministic', suffix='.det_noacc_vmodule')
matrix_original = matrix_original.dataMatrix

mae = lib.MAE(matrix_interpolada, matrix_original)
print(mae)
