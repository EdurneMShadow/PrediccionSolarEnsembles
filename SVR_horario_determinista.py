# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:07:02 2017

@author: Edurne
"""

#get_ipython().magic('matplotlib inline')
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import time
import bisect

import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer


#Carga de datos
df = pd.read_csv('w_e.csv', sep=',')

print("nFilas: %d\tnColumnas: %d\n" % (df.shape[0], df.shape[1]) )
print("Columnas:\n", np.array(df.columns))

l_features = list( df.columns[2 : ] )
target = df.columns[1]

#Conjunto de entrenamiento y test
train = df.ix[0:3945]
test = df.ix[3946:]

#Modelo SVR
X_train = train[l_features]
y_train = train[target]
X_test = test[l_features]
y_test = test[target]

clf = svm.SVR()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

print("MAE: ", abs(y_test - y_predict).mean())
#plt.scatter(X_test['U10'],y_test, label='data')
#plt.plot(X_test['U10'], y_predict, 'darkorange')

#Modelo SVR con kernel
svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = svm.SVR(kernel='linear', C=1e3)
svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
print("MAE: ", abs(y_test - y_rbf).mean())

y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
print("MAE: ", abs(y_test - y_lin).mean())

y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
print("MAE: ", abs(y_test - y_poly).mean())

















