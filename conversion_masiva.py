#!/usr/bin/env python3
# encoding: utf-8

import sys
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

import glob
import pandas as pd
import numpy as np
import subprocess
import DataMatrix_NWP as dm
import datetime
import desagregar as lib

SCRIPT = "/gaa/home/edcastil/scripts/netcdf_to_myp.py"

for i in glob.glob("/gaa/home/edcastil/scripts/conversion/nc/*.nc"):
    program_args = (["python", SCRIPT, i])
    print(" ".join(program_args))
    subprocess.check_call(program_args)
    
for i in glob.glob("/gaa/home/edcastil/scripts/conversion/myp/*"):
    fecha = i[3] #TODO: ver cómo sacar la fecha del nombre del fichero
    f = datetime.datetime.strptime(fecha, '%Y%m%d')
    matrix_control_dia = (dm.DataMatrix(f,
    '/gaa/home/edcastil/scripts/conversion/myp','/gaa/home/edcastil/scripts/conversion/datamatrix',
    ifexists=False,model='ensembles',n_ens=1, suffix='.control', tags=dm.nwp_ensembles_tags, delta=1))
    matrix_control_dia.save_matrix(suffix = '.control')
    
#MATRIZ 2013
inicio = datetime.datetime(2013,1,1)
fin = datetime.datetime(2013,12,31)
index = lib.crear_indice_anio(inicio,fin)
dias = [str(i) for i in index]

matrix_2013 = pd.DataFrame()

for i in dias:
    matrix_control = (dm.DataMatrix(datetime.datetime.strptime(i, '%Y%m%d'),
    '/gaa/home/edcastil/scripts/conversion/datamatrix','/gaa/home/edcastil/scripts/conversion/datamatrix',
    ifexists=True,model='ensembles',n_ens = 1, suffix='.control', tags = dm.nwp_ensembles_tags))
    
    matrix_control = matrix_control.dataMatrix
    matrix_2013 = pd.concat([matrix_2013, matrix_control])
    
lib.guardar_matriz(matrix_2013, index[-1], '.control')
    
    
#MATRIZ 2014
inicio = datetime.datetime(2014,1,1)
fin = datetime.datetime(2014,12,31)
index = lib.crear_indice_anio(inicio,fin)
dias = [str(i) for i in index]

matrix_2014 = pd.DataFrame()

for i in dias:
    matrix_control = (dm.DataMatrix(datetime.datetime.strptime(i, '%Y%m%d'),
    '/gaa/home/edcastil/scripts/conversion/datamatrix','/gaa/home/edcastil/scripts/conversion/datamatrix',
    ifexists=True,model='ensembles',n_ens = 1, suffix='.control', tags = dm.nwp_ensembles_tags))
    
    matrix_control = matrix_control.dataMatrix
    matrix_2014 = pd.concat([matrix_2014, matrix_control])
    
lib.guardar_matriz(matrix_2014, index[-1], '.control')    
    
    
    
    
    
    
    
    