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
    fecha = i[42:][:-8]
    f = datetime.datetime.strptime(fecha, '%Y%m%d')
    matrix_control_dia = (dm.DataMatrix(f,'/gaa/home/edcastil/scripts/conversion/myp/',
    '/gaa/home/edcastil/scripts/conversion/datamatrix/',ifexists=False,model='ensembles',
    n_ens=1, suffix='.control', tags=dm.nwp_ensembles_tags, delta=0))
    matrix_control_dia.save_matrix(suffix = '.control')
    
#MATRIZ 2013
matrix_control = (dm.DataMatrix(datetime.datetime(2013,12,31),'/gaa/home/edcastil/scripts/conversion/myp/','/gaa/home/edcastil/scripts/conversion/',ifexists=True,model='ensembles',n_ens = 1, suffix='.control', tags = dm.nwp_ensembles_tags, delta = 364))
matrix_control.save_matrix(suffix=".control")
#MATRIZ 2014
matrix_control = (dm.DataMatrix(datetime.datetime(2014,12,31),'/gaa/home/edcastil/scripts/conversion/myp/','/gaa/home/edcastil/scripts/conversion/',ifexists=True,model='ensembles',n_ens = 1, suffix='.control', tags = dm.nwp_ensembles_tags, delta = 364))
matrix_control.save_matrix(suffix=".control")