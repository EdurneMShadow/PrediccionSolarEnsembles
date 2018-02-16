#!/usr/bin/env python3
# encoding: utf-8
"""
Created on Thu Feb  8 05:49:41 2018

@author: Edurne
"""
import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')
import pandas as pd
import numpy as np
import desagregar as lib

nombre_parcial = '/gaa/home/edcastil/datos/20151231.mdata.ens_'

for i in range(50):
    nombre = nombre_parcial + str(i) + '.csv'
    data = pd.read_csv(nombre, index_col=0)
    nuevo = lib.desagregar_control(data)
    nuevo.to_csv('/gaa/home/edcastil/datos/20151231.mdata.ens_desacc_' + str(i) + '.csv')