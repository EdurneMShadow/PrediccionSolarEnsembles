import DataMatrix_NWP as dm
import datetime
import pandas as pd
import numpy as np
import sunrise as sr
import libdata as ut

def guardar_matriz(matriz, fecha, sufijo=''):
    nombre_fichero = '/scratch/gaa/edcastil/'+ str(fecha) + '.mdata' + sufijo
    datos = np.hstack([matriz.index[:,np.newaxis], matriz.values])
    columnas = matriz.columns
    np.save(nombre_fichero, datos)
    print('Matriz guardada')
    nombre_fichero_columnas = nombre_fichero + '.columns'
    np.save(nombre_fichero_columnas,columnas)

def determinista_a_trihorario_acc():
    # Estas líneas pasan el modelo determinista horario acumulado a trihorario acumulado.+
    tags = ['FDIR', 'CDIR','SSRD', 'SSR', 'SSRC']
    matrix=dm.DataMatrix(datetime.datetime(2015,12,31),'/scratch/gaa/data/solar_ecmwf/','/scratch/gaa/data/solar_ecmwf/',ifexists=True,model='deterministic',suffix='.det_noacc_vmodule')
    #matrix.dataMatrix #He creado un objeto de la clase DataMatrix, que tiene un atributo dataMatrix.
    #matrix.dataMatrix.columns #Accede a las columnas de la matriz
    m = matrix.dataMatrix
    land_grid = dm.select_pen_grid() #Selecciona solo las longitudes y latitudes de España.

    pen_cols = matrix.query_cols(latlons=land_grid,tags=['FDIR','SSR','SSRC','CDIR','SSRD']) #Estoy seleccionando solo las variables de radiacion (array)
    rad_var = matrix.dataMatrix[pen_cols] #submatriz solo con las columnas de las variables de radiación.

    #Agregar: sumar una fila con la anterior. En la última fila se tendrá la acumulación de todas las anteriores
    dia = 1
    for i in range(len(rad_var)-1):
        print(dia)
        if dia<24:
            rad_var.values[i+1] = rad_var.values[i+1] + rad_var.values[i]
            dia+=1
        else:
            dia = 1


    dates = pd.date_range('20150101','20160101',freq='3H')[:-1]
    index = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])#lista de indices

    rad_var_3h = rad_var.loc[index] #extraer el conjunto trihorario que queremos de la matriz
    rad_var_3h = rad_var_3h.diff() #diferencia: un elemento menos el anterior (por filas)

    index_day = sr.filter_daylight_hours(index)

    # nights=[]
    # for i in index:
    #     if i not in index_day:
    #         nights.append(i)

    index_night = [i for i in index if i not in index_day]

    rad_var_3h.loc[index_night] = 0 #sobrescribe en la matriz actual
    fecha = matrix.date.strftime(matrix.date_format)
    guardar_matriz(rad_var_3h, fecha, sufijo=".det_3h_acc")
    return rad_var_3h

'''Estas líneas pasan el clear-sky horario no acumulado a horario acumulado.'''
def CS_a_acumulado():
    df = pd.DataFrame()
    cs = ut.load_CS(df, shft=0)

    #Seleccionar solo las columnas de la península
    #NOTE: No hace falta filtrar el CS por península, porque al cargarlo con shft=0
    #      las columnas recuperadas coinciden con las de la península
    # rejilla_peninsula = dm.select_pen_grid()
    # cs_peninsula = pd.DataFrame()
    # for i in rejilla_peninsula:
    #     col = '(' + str(i[1]) + ', ' + str(i[0]) + ')' +  ' CS H'
    #     cs_peninsula[col] = cs[col]

    #Agregar
    dia = 0
    for i in range(len(cs)):
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
    cs_3h = cs_3h.diff()

    #Poner horas de noche a cero
    index_day = sr.filter_daylight_hours(index)
    cs_3h.loc[index_night] = 0 #sobrescribe en la matriz actual
    cs_3h.to_csv('cs_3h_acc.csv')

def interpolacion():
    dates = pd.date_range('20150101','20160101',freq='3H')[:-1]
    index = np.array([int(d.strftime("%Y%m%d%H")) for d in dates])#lista de indices
    cs_h = cs.loc[index]

    r_pred = pd.DataFrame(index=cs_h.index, columns=rad_var_3h.columns)

    var = 0
    for i in cs_h.index:
        cs_horario = cs_h.loc[i]
        terminacion = str(i)[8:]
        if int(terminacion) > var:
            if var == 21:
                var = 0
            else:
                var+=3
        if len(str(var)) < 2:
            var_string = '0' + str(var)
        else:
            var_string = str(var)
        indice = int(str(i)[:8] + var_string)
        cs_acc = cs_3h.loc[indice]
        rad_acc = rad_var_3h.loc[indice]
        fila = cs_horario/cs_acc*rad_acc
        r_pred.loc[indice] = fila
