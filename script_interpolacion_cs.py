# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 13:06:28 2017

@author: Edurne
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



cs = pd.read_csv('./Prueba_interpolacion/matriz_punto_cs.csv', index_col = 0)
cs3 = pd.read_csv('./Prueba_interpolacion/matriz_punto_cs3.csv', index_col = 0)
m_original = pd.read_csv('./Prueba_interpolacion/matriz_punto_original.csv', index_col = 0)
m_trihoraria = pd.read_csv('./Prueba_interpolacion/matriz_punto_trihoraria.csv', index_col = 0)

def extension_horaria(var_acc_3h):
    """Giver  hourly values of a variable with 8 values accumulated every three hours.
    The values at hours h-2, h-1, h for 0, 3, 6, 9, 12, 15, 18 and 21 will be 
    the same.
    It is used in order to apply a CS interpolation of the form 
    var_acc_int = var_acc_h * cs_coef
    """
    try:
        assert len(var_acc_3h) == 8, "argument must have 8 values"
        
        #drop first 0 and add last 0. to work with 8 consecutive 3-h groups
        l_var_acc_3h = list(var_acc_3h[1 : ]) + [0.] 
        
        var_acc_h = []
        for h in range(24):
            var_acc_h.append(l_var_acc_3h[h//3])
            
        #add first 0. and drop last 0.
        return np.array([0.] + var_acc_h[ : 23])
    
    except AssertionError:
        print("argument must have 8 values")
        
def extension_horaria_cs(cs_3h):
    """Computes the denominator cs_acc in order to apply a CS interpolation of the form 
    var_acc_int = var_acc * cs / cs_acc.
    
    cs can be any variable although 'CS H' is expected.
    """
    cs_acc_h  = extension_horaria(cs_3h)
    cs_acc_h[ cs_acc_h == 0. ] = 1.

    return cs_acc_h


dates = pd.date_range('20150101','20160101',freq='1H')[:-1]
index = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])