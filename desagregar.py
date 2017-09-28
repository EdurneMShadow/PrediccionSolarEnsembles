import DataMatrix_NWP as dm
import datetime
import pandas as pd
import numpy as np
import sunrise as sr
import libdata as ut
from scipy.interpolate import CubicSpline

#%load_ext autoreload
#%autoreload 2

'''Ejemplos de cargas de matrices'''
#matrix_tri=dm.DataMatrix(datetime.datetime(2015,12,31),'/gaa/home/edcastil/','/gaa/home/edcastil/',ifexists=True,model='deterministic',suffix='.det_3h_acc', tags = ['FDIR', 'CDIR','SSRD', 'SSR', 'SSRC'], latlons = latlon)

#matrix_original = dm.DataMatrix(datetime.datetime(2015,12,31), '/gaa/home/data/solar_ecmwf/', '/gaa/home/data/solar_ecmwf/', ifexists = True, model='deterministic', suffix='.det_noacc_vmodule')
#latlon = dm.select_pen_grid()
#matrix_interpolada = dm.DataMatrix(datetime.datetime(2015,12,31), '/gaa/home/edcastil/', '/gaa/home/edcastil/', ifexists = True, model='deterministic', suffix='.det_interpolado_cs', tags = ['FDIR', 'CDIR','SSRD', 'SSR', 'SSRC'], latlons = latlon)

def guardar_matriz(matriz, fecha, sufijo=''):
    '''Este método crea dos ficheros .npy, uno con los datos y otro con los nombres de las columnas de la matriz que se
    le haya pasado por parámetro. La fecha y el sufijo se usan para la creación del nombre del fichero según el formato:
    fecha_ultima_hora.mdata.sufijo y fecha_ultima_hora.mdata.sufijo.columns para el fichero de las columnas.'''

    nombre_fichero = '/gaa/home/edcastil/'+ str(fecha) + '.mdata' + sufijo
    datos = np.hstack([matriz.index[:,np.newaxis], matriz.values])
    columnas = matriz.columns
    np.save(nombre_fichero, datos)
    print('Matriz guardada')
    nombre_fichero_columnas = nombre_fichero + '.columns'
    np.save(nombre_fichero_columnas,columnas)

def seleccionar_punto_rejilla(matrix, tags, latlon):
    '''Dada una matriz, el punto que se quiere obtener y las tags asociadas a ese punto, se devuelve
       la submatriz correspondiente.
    '''
    columnas = []
    for i in tags:
        columnas.append(latlon + ' ' + str(i))
    return matrix[columnas]

def determinista_a_trihorario_acc(radiacion = False):
    '''Este método lee la matriz determinista horaria desacumulada que se encuentra en data/solar_ecmwf y la transforma en trihoraria
    acumulada. Se ha usado para comprobar que la interpolación mediante clear-sky es factible. Lo que se ha hecho es:
    - Seleccionar las longitudes y latidudes de la península, así como las variables de radiación que queremos interpolar.
    - Acumular la matriz sumando las filas de 24 en 24.
    - Seleccionar los índices trihorarios.
    - Seleccionar las horas de noche y ponerlas a cero.
    Finalmente se devuelve la nueva matriz, además de guardarla en un fichero .npy en edcastil'''

    matrix=dm.DataMatrix(datetime.datetime(2015,12,31),'/gaa/home/data/solar_ecmwf/','/gaa/home/data/solar_ecmwf/',ifexists=True,model='deterministic',suffix='.det_noacc_vmodule')
    m = matrix.dataMatrix
    land_grid = dm.select_pen_grid() #Selecciona solo las longitudes y latitudes de España.

    if(radiacion is True):
        pen_cols = matrix.query_cols(latlons=land_grid,tags=['FDIR','SSR','SSRC','CDIR','SSRD']) #Estoy seleccionando solo las variables de radiacion (array)
        submatrix = matrix.dataMatrix[pen_cols] #submatriz solo con las columnas de las variables de radiación.
        sufijo = ".det_3h_acc"
    else:
        pen_cols = matrix.query_cols(latlons=land_grid,tags=['FDIR','SSR','SSRC','CDIR','SSRD', ''])
        submatrix = m
        sufijo = ".det_3h_acc_entera"

    #Agregar: sumar una fila con la anterior. En la última fila se tendrá la acumulación de todas las anteriores
    dia = 1
    for i in range(len(submatrix)-1):
        if dia<24:
            submatrix.values[i+1] = submatrix.values[i+1] + submatrix.values[i]
            dia+=1
        else:
            dia = 1

    #Selección de los índices trihorarios
    dates = pd.date_range('20150101','20160101',freq='3H')[:-1]
    index = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])#lista de indices

    submatrix_3h = submatrix.loc[index] #extraer el conjunto trihorario que queremos de la matriz
    submatrix_3h = submatrix_3h.diff() #diferencia: un elemento menos el anterior (por filas)

    #Poner las horas de noche a cero.
    index_day = sr.filter_daylight_hours(index)
    index_night = [i for i in index if i not in index_day]

    submatrix_3h.loc[index_night] = 0 #sobrescribe en la matriz actual
    fecha = matrix.date.strftime(matrix.date_format)
    guardar_matriz(submatrix_3h, fecha, sufijo=sufijo)
    return submatrix_3h


def CS_a_acumulado():
    '''Este método pasa el clear-sky horario no acumulado a trihorario acumulado. El procedimiento seguido es el mismo
    que en el método anterior.'''

    df = pd.DataFrame()
    cs = ut.load_CS(df, shft=0)

    #Seleccionar solo las columnas de la península
    #No hace falta filtrar el CS por península, porque al cargarlo con shft=0
    #      las columnas recuperadas coinciden con las de la península
    # rejilla_peninsula = dm.select_pen_grid()
    # cs_peninsula = pd.DataFrame()
    # for i in rejilla_peninsula:
    #     col = '(' + str(i[1]) + ', ' + str(i[0]) + ')' +  ' CS H'
    #     cs_peninsula[col] = cs[col]

    #Agregar
    dia = 1
    for i in range(len(cs)-1):
        if dia<24:
            cs.values[i+1]=cs.values[i+1] + cs.values[i]
            dia+=1
        else:
            dia = 1

    #Coger de 3 en 3h
    dates= pd.date_range('20150101','20160101',freq='3H')[:-1]
    index=np.array([int(d.strftime("%Y%m%d%H")) for d in dates])
    cs_3h = cs.loc[index]
    cs_3h = cs_3h.diff()

    #Poner horas de noche a cero
    index_day = sr.filter_daylight_hours(index)
    index_night = [i for i in index if i not in index_day]
    cs_3h.loc[index_night] = 0 #sobrescribe en la matriz actual
    cs_3h.to_csv('cs_3h_acc.csv')
    return cs_3h

def interpolacion(cs, cs_3h, r_3h):
    '''Calcula la matriz horaria a partir de la trihoraria mediante el clear-sky. La fórmula aplicada es:
       r_1h = cs_1h/cs_3h*r_3h
    '''

    latlon = dm.select_pen_grid()
    pen_cols = dm.query_cols(latlons=latlon,tags=['CS H'])
    not_pen_cols = [i for i in cs.columns if i not in pen_cols]
    cs_3h = cs_3h.drop(not_pen_cols,1)
    cs = cs.drop(not_pen_cols,1)

    cs_3h = pd.concat([cs_3h, cs_3h, cs_3h])
    cs_3h = cs_3h.sort_index()
    cs_3h = pd.concat([cs_3h[2:], cs_3h[:2]]) #pone las dos primeras filas al final

    dates = pd.date_range('20150101','20160101',freq='1H')[:-1]
    index1 = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])
    cs_3h.index = index1

    cs = cs[17520:]

    division = cs/cs_3h

    index_day = sr.filter_daylight_hours(index1)
    index_night = [i for i in index1 if i not in index_day]
    division.loc[index_night] = 0

    #arreglar infinitos

    for i in index1:
        terminacion = str(i)[8:]
        if terminacion == '16':
            division.loc[i] = 0
        if terminacion == '17':
            division.loc[i] = 0



    division_extendida = pd.concat([division,division,division,division,division], axis=1)
    division_extendida.columns = r_3h.columns

    r_3h = pd.concat([r_3h, r_3h, r_3h])
    r_3h = r_3h.sort_index()
    r_3h = pd.concat([r_3h[2:], r_3h[:2]]) #pone las dos primeras filas al final

    r_3h.index = index1

    interpolado = division_extendida*r_3h

    fecha = '2015123100'
    guardar_matriz(interpolado, fecha, sufijo=".det_interpolado_cs")
    return interpolado


def MAE(df_interpolado, df_original):
    '''Este método calcula tanto el MAE de cada columna como el MAE global dados dos conjuntos a comparar.
    Para el cálculo no se tienen en cuenta las horas de noche. Procedimiento:
        - En primer lugar se calcula la resta absoluta entre los dos df, dando un valor para cada posición de la tabla.
        - Después se suman esos valores por columnas y se divide por el n_filas para tener un valor de mae para cada columna.
        - Finalmente se hace la media de todos los maes para tener el mae global.
        Se devuelve un diccionario con los dos maes.
    '''
    index = df_original.index
    latlon = dm.select_pen_grid()
    hours_day = sr.filter_daylight_hours(index)
    hours_night = [i for i in index if i not in hours_day]

    df_interpolado = df_interpolado.drop(hours_night)
    df_original = df_original.drop(hours_night)

    pen_cols = dm.query_cols(latlons=latlon,tags=['FDIR','SSR','SSRC','CDIR','SSRD'])
    not_pen_cols = [i for i in df_original.columns if i not in pen_cols]
    df_original = df_original.drop(not_pen_cols,1)
    columnas = df_original.columns

    resta_absoluta = np.abs(df_original - df_interpolado)
    mae_columnas = resta_absoluta.mean()
    mae_global = mae_columnas.mean()

    return mae_columnas, mae_global

def interpolacion_cubica(matrix, indice_1h, indice_3h):
    f = CubicSpline(indice_3h, matrix.loc[indice_3h])
    return f(indice_1h) 

def MAE_diario(df_interpolado, df_original):
    '''Devuelve el MAE calculado para cada día de la matriz que se pasa por parámetro.'''

    inicio_dia = 0
    fin_dia = 24
    mae_dias = {}

    while fin_dia < len(df_original):
        original = df_original[inicio_dia:fin_dia]
        interpolado = df_interpolado[inicio_dia:fin_dia]

        hours_day = sr.filter_daylight_hours(original.index)
        hours_night = [i for i in original.index if i not in hours_day]
        interpolado = interpolado.drop(hours_night)
        original = original.drop(hours_night)

        resta_absoluta = np.abs(original-interpolado)
        mae_dias[original.index[0]] = resta_absoluta.mean()
        inicio_dia = fin_dia
        fin_dia += 24
    return mae_dias

def obtener_dia_completo(fecha):
    '''Dada una fecha en formato YYYYMMDDHH entero, se devuelve una lista con todas las horas que conforman dicho día.
       Dicho de otra forma, se devuelve un índice para un día.
    '''
    dia = str(fecha)[:8]
    indice = []
    for i in range (24):
        if len(str(i)) < 2 :
            hora = '0' + str(i)
        else:
            hora = str(i)
        indice.append(int(dia + hora))
    return indice

def crear_indice_anio(inicio, fin):
    '''Dados dos datetime, devuelve el índice con todas las horas entre ellos'''
    index = []
    delta = fin - inicio
    for i in range (0,int(delta.total_seconds()), 3600):
        index.append(int((inicio + timedelta(hours = i/3600)).strftime("%Y%m%d%H")))
    return index

#prodsTotal = pd.read_csv('/gaa/home/alecat/data/prodsTotal.csv', index_col=0)
#index_train = crear_indice_anio(datetime.datetime(2013,1,1), datetime.datetime(2014,1,1))
#index_val = crear_indice_anio(datetime.datetime(2014,1,1), datetime.datetime(2015,1,1))
#index_test = crear_indice_anio(datetime.datetime(2015,1,1), datetime.datetime(2016,1,1))
#
#prod_train = prodsTotal.loc[index_train]
#prod_val = prodsTotal.loc[index_val]
#prod_test = prodsTotal.loc[index_test]