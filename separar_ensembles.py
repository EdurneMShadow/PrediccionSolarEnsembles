# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:17:12 2018

@author: Edurne
"""

import pandas as pd
import numpy as np

data = pd.read_csv('/gaa/home/edcastil/datos/20151231.mdata.ens.csv', index_col=0)
columnas = data.columns

n = 0
for i in range(0, 208800, 4176):
    c = columnas[i : i + 4176]
    d = data[c]
    d.to_csv('/gaa/home/edcastil/datos/20151231.mdata.ens_' + str(n) + '.csv')
    n += 1