import DataMatrix_NWP as dm
import datetime
import pandas as pd
import numpy as np
import sunrise as sr
import utilities as ut

def determinista_a_trihorario_acc():
    # Estas líneas pasan el modelo determinista horario acumulado a trihorario acumulado.
    matrix=dm.DataMatrix(datetime.datetime(2015,12,31),'/scratch/gaa/data/solar_ecmwf/deterministic/myp/','/scratch/gaa/data/solar_ecmwf/',ifexists=True,model='deterministic',suffix='.det')
    matrix.dataMatrix #He creado un objeto de la clase DataMatrix, que tiene un atributo dataMatrix.
    matrix.dataMatrix.columns #Accede a las columnas de la matriz

    land_grid=dm.select_pen_grid() #Selecciona solo las longitudes y latitudes de España.

    pen_cols=matrix.query_cols(latlons=land_grid,tags=['FDIR','SSR','SSRC','CDIR','SSRD']) #Estoy seleccionando solo las variables de radiacion (array)
    rad_var=matrix.dataMatrix[pen_cols] #submatriz solo con las columnas de las variables de radiación.
    dates= pd.date_range('20150101','20160101',freq='3H')[:-1]
    index=np.array([int(d.strftime("%Y%m%d%H")) for d in dates])#lista de indices
    rad_var_3h=rad_var.loc[index] #extraer el conjunto trihorario que queremos de la matriz
    rad_var_3h=rad_var_3h.diff() #diferencia: un elemento menos el anterior (por filas)

    index_day=sr.filter_daylight_hours(index)

    # nights=[]
    # for i in index:
    #     if i not in index_day:
    #         nights.append(i)

    index_night = [i for i in index if i not in index_day]

    rad_var_3h.loc[index_night] = 0 #sobrescribe en la matriz actual

'''Estas líneas pasan el clear-sky horario no acumulado a horario acumulado.'''
def CS_a_acumulado():
    df = pd.DataFrame()
    cs = ut.load_CS(df)
    #Seleccionar solo las columnas de la península
    rejilla_peninsula = dm.select_pen_grid()
    columnas_cs = []
    #TODO: preguntar por cómo sacar la peninsula.
    #Última situación: rejilla_peninsula + CS H no es ninguna columna
    for i in rejilla_peninsula:
        col = str(i)+' CS H'
        columnas_cs.append(col)

    #Agregar
    dia = 0
    #NOTE: en el fichero cs la primera hora es uno y no cero.
    for i in range(len(cs)): #TODO: cambiar cs por el nombre de la variable de la peninsula
        if dia<24:
            if i<len(cs)-1:
                cs.values[i+1]=cs.values[i+1] + cs.values[i]
                dia+=1
        else:
            dia = 0
    #Coger de 3 en 3h
    dates= pd.date_range('20150101','20160101',freq='3H')[:-1]
    index=np.array([int(d.strftime("%Y%m%d%H")) for d in dates])
    cs_3h = cs.loc[index]
