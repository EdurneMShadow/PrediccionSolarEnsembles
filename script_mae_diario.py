#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import pandas as pd
import desagregar as lib


r_original = pd.read_csv('./Plots/radiacion_original.csv', index_col=0)
r_interpolado = pd.read_csv('./Plots/radiacion_interpolada.csv', index_col=0) 
tags = ['FDIR','CDIR','SSR','SSRC','SSRD']

mae = lib.MAE_diario(r_interpolado[r_interpolado.columns[0]], r_original[r_original.columns[0]]) 
maximo_error = max(mae, key=lambda i: mae[i])

mae01 = lib.MAE_diario(r_interpolado[r_interpolado.columns[1]], r_original[r_original.columns[1]]) 
maximo_error01 = max(mae01, key=lambda i: mae01[i])

mae02 = lib.MAE_diario(r_interpolado[r_interpolado.columns[2]], r_original[r_original.columns[2]]) 
maximo_error02 = max(mae02, key=lambda i: mae02[i])

mae03 = lib.MAE_diario(r_interpolado[r_interpolado.columns[3]], r_original[r_original.columns[3]]) 
maximo_error03 = max(mae03, key=lambda i: mae03[i])

mae04 = lib.MAE_diario(r_interpolado[r_interpolado.columns[4]], r_original[r_original.columns[4]]) 
maximo_error04 = max(mae04, key=lambda i: mae04[i])

print('fdir: ' + str(maximo_error))
print('cdir: ' + str(maximo_error01))
print('ssr: ' + str(maximo_error02))
print('ssrc: ' + str(maximo_error03))
print('ssrd: ' + str(maximo_error04))
