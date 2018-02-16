#!/usr/bin/env python3
# encoding: utf-8

"""
Created on Sun Feb  4 22:47:28 2018

@author: Edurne
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

for i in range(50):
    fichero = '/gaa/home/edcastil/datos/20151231.mdata.ens_' + str(i) + '.csv'
    data = pd.read_csv(fichero, index_col=0)
    a = pd.concat([data.loc[data.index[:174]], data.loc[data.index[186:]]])
    a.to_csv(fichero)