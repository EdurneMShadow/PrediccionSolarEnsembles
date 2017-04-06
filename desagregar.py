import DataMatrix_NWP as dm
import datetime
import pandas as pd
import numpy as np
import sunrise as sr
import utilities as ut

def load_CS(df, shft=1):
     cs_file = '/scratch/gaa/alecat/data/clear_sky/cs_15min.npy'
     cs_columns_file = '/scratch/gaa/alecat/data/clear_sky/cs_15min_cols.npy'
     data = np.load(cs_file)
     columns = np.load(cs_columns_file)
     index = data[:, 0].astype(int)
     df_cs = pd.DataFrame(data[:, 1:], columns=columns, index=index)
     if shft > 0:
         df_cs_k = shift_df(df_cs, shft)
         df_cs_k.columns = list(map(lambda c: c + '-K', df_cs_k.columns))
         df_cs_total = pd.concat([df_cs, df_cs_k], axis=1, join='inner')
     else:
         df_cs_total = df_cs

     if not df.empty:
         return df.join(df_cs_total)
     return df_cs_total

#-------------------------------------------------------------------------------

# Estas líneas pasan el modelo determinista horario acumulado a trihorario.
matrix=dm.DataMatrix(datetime.datetime(2015,12,31),'/scratch/gaa/alecat/data/ecmwf/deterministic/myp/','/scratch/gaa/alecat/data/ecmwf/',ifexists=True,model='deterministic',suffix='.det')
matrix.dataMatrix #He creado un objeto de la clase DataMatrix, que tiene un atributo dataMatrix.
matrix.dataMatrix.columns #Accede a las columnas de la matriz

land_grid=dm.select_pen_grid() #Selecciona solo las longitudes y latitudes de España.

pen_cols=matrix.query_cols(latlons=land_grid,tags=['FDIR','SSR','SSRC','CDIR','SSRD']) #Estoy seleccionando solo las variables de radiacion (array)
rad_var=matrix.dataMatrix[pen_cols] #submatriz solo con las columnas de las variables de radiación.
dates= pd.date_range('20150101','20160101',freq='3H')[:-1]
index=np.array([int(d.strftime("%Y%m%d%H")) for d in dates])#lista de indices
rad_var_3h=rad_var.loc[index] #extraer el conjunto trihorario que queremos de la matriz
rad_var_3h=rad_var_3h.diff()

index_day=sr.filter_daylight_hours(index)

# nights=[]
# for i in index:
#     if i not in index_day:
#         nights.append(i)

index_night = [i for i in index if i not in index_day]

rad_var_3h.loc[index_night] = 0 #sobrescribe en la matriz actual


#Estas líneas pasan el clear-sky horario no acumulado a trihorario.

cs = load_CS()

#Interpolar el determinista trihorario a horario.

cs = pd.read_csv('/scratch/gaa/alecat/data/2015123100.cs_3h.csv')
