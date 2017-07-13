# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:22:05 2017

@author: Edurne
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn import svm

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

import DataMatrix_NWP as dm
import datetime

#Carga de datos
matrix_original = dm.DataMatrix(datetime.datetime(2015,12,31), '/gaa/home/data/solar_ecmwf/', '/gaa/home/data/solar_ecmwf/', ifexists = True, model='deterministic', suffix='.det_noacc_vmodule')
matrix = matrix_original.dataMatrix

#Escalar los datos
########## saling data using StandardScaler
standard_scaler = StandardScaler()
data_t = standard_scaler.fit_transform(matrix)
print ('Media: ' + str(data_t.mean()))
print ('Varianza: ' + str(data_t.var()))


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

########## CV estimation of generalization performance using #cross_val_score using the default Ridge model
X = data_t
y = df[target]
ridge = Ridge ()
mae = make_scorer(mean_absolute_error, greater_is_better=True)
scores = cross_val_score(ridge, X, y, cv=9, scoring=mae)
print(scores) 

########## hyperparameter tuning by CV using KFold and GridSearchCV