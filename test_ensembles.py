#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

assert (len(sys.argv) >= 2), 'Debe haber un argumento'
n_ensemble = str(sys.argv[1])

#########Carga del dataset de control y el ensemble original##########

df_control = pd.read_csv('/gaa/home/edcastil/datos/control/20131231.mdata.control_desagregado_resolucion.csv', index_col=0)
#df_ens = pd.read_csv('/gaa/home/edcastil/datos/ensembles_nuevos/20151231.mdata.ens_desacc_1.csv', index_col=0)
#df_ens = pd.read_csv('/gaa/home/alecat/edurne_ens/dfs/2015123100.mdata.ens_1_noacc.csv', index_col=0)

name = '/gaa/home/alecat/edurne_ens/dfs/2015123100.mdata.ens_1.csv'

df_ens = pd.read_csv(name, index_col=0)

#########Preprocesado del ensemble para que se pueda usar junto con el control##########
ens_cols = df_ens.columns
control_cols = sorted(df_control.columns)

def from_ens_2_control_var_names(s):
    pos_ini = s.find('(')
    pos_med = s.find(',')
    pos_fin = s.find(')')
    
    #return s[pos_ini : pos_fin+1] + ' ' + s[ : pos_ini].upper()
    return '(' + s[pos_med+2 : pos_fin] + ', ' + s[pos_ini+1 : pos_med] + ') ' + s[ : pos_ini].upper()

ens_cols_2 = []

for var in ens_cols:
    ens_cols_2.append( from_ens_2_control_var_names(var) )

df_ens_2 = pd.DataFrame(data=df_ens.values, index=df_ens.index, columns=ens_cols_2)

cols_both = list( set(control_cols).intersection(set(ens_cols_2)))

#########Desagregación del ensemble##########
#def desagregar_control(df):
#    tags = ['fdir', 'cdir','ssrd', 'ssr']    
#    for j in df.columns:
#        for k in tags:
#            if j.find(k) is not -1:
##        var = j[-7:].strip()
##        var = var[:-2].strip()
##        var = var.replace(') ', '')
#                new_column = []
#                for i in range (0,df.shape[0],6):        
#                    index = df.index[i:i+6]
#                    submatrix = df[j].loc[index]
#                    new_index = np.arange(6)
#                    submatrix.index = new_index
#                    new_column.append(submatrix.loc[0])
#                    for h in range(1,6):      
#                    #for h in range(1,7):
#                        new_column.append(submatrix.loc[h] - submatrix.loc[h-1])
#                df[j] = new_column
#
#    return df
#df_ens = desagregar_control(df)

df_ens_desacc = pd.DataFrame()
idx_6 = df_ens.index % 100 == 6

for var in control_cols:
    radiation_var = False
    for name in ['CDIR', 'FDIR', 'SSR', 'SSRD']:
        if name in var:
            df_ens_desacc[var] = df_ens_2[var].diff()
            df_ens_desacc[var].loc[idx_6] = df_ens_2[var].loc[idx_6]
            radiation_var = True
            break
    if not radiation_var:
        df_ens_desacc[var] = df_ens_2[var]

df_ens_desacc.to_csv('/gaa/home/edcastil/datos/ens_desacc/20151231.mdata.ens_1_desacc.csv')

#############Predicciones################
df_y_2013 = pd.read_csv('/gaa/home/edcastil/datos/Prod_2013_trihorario.csv', index_col=0)
df_y_2015 = pd.read_csv('/gaa/home/edcastil/datos/Prod_2015_trihorario.csv', index_col=0)

df_control_prod = pd.concat([df_control[ sorted(cols_both) ], df_y_2013],  join='inner', axis=1)
df_ens_2_prod   = pd.concat([df_ens_desacc[ sorted(cols_both) ], df_y_2015], join='inner', axis=1)

l_vars =  sorted(df_control_prod.columns[ : -1])
target =  df_control_prod.columns[-1]

sc = MinMaxScaler()

x_train = sc.fit_transform(df_control_prod[l_vars].values.astype(float))
x_test  = sc.transform(df_ens_2_prod[l_vars].values.astype(float))

clave = (100.0, 0.15473252272286195, 0.0009578544061302681)
svr = SVR(C=clave[0], gamma=clave[2], epsilon=clave[1], kernel='rbf', shrinking = True, tol = 1.e-6)
svr.fit(x_train,df_control_prod[target])
#y_train_pred = svr.predict(x_train_escalado)
y_pred = svr.predict(x_test)
mae = mean_absolute_error(df_ens_2_prod[target], y_pred)

lista_predicciones = [df_ens_2_prod[target], y_pred]
nombre = '/gaa/home/edcastil/scripts/resultados_control/comparaciones_svr_test_ens_' + n_ensemble + '.pkl'
pickle.dump(lista_predicciones, open(nombre, 'wb' ))

nombre = '/gaa/home/edcastil/scripts/resultados_ensembles/resultados_test_ensemble_' + n_ensemble + '.txt'
f = open(nombre, 'w')
f.write(str(clave) + '\n')
f.write('Error de test: ' + str(mae) + '\n')
f.close()