#!/usr/bin/env python3
# encoding: utf-8

"""
Created on Sun Feb  4 22:47:28 2018

@author: Edurne
"""
import subprocess
import sys
import numpy as np
import pandas as pd
import time as t
sys.path.insert(0, '/gaa/home/edcastil/scripts/')

nombre_parcial = '/gaa/home/edcastil/datos/20151231.mdata.ens_'
SCRIPT = "/gaa/home/edcastil/scripts/SVR_ensembles.py"

def main():
    for i in range (50):
        nombre = nombre_parcial + 'desacc_' + str(i) + '.csv'
        n_ensemble = str(i)
        program_args = (["sbatch", "-A", "gaa_serv", "-p", "gaa", "--mail-type=ALL", 
				"--mail-user=edurne.castillo@estudiante.uam.es", "--mem-per-cpu=50000", SCRIPT, nombre, n_ensemble])
        print(" ".join(program_args))
        subprocess.check_call(program_args)

if __name__ == "__main__":
	main()