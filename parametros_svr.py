#!/usr/bin/env python3
# encoding: utf-8

import subprocess
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/gaa/home/edcastil/scripts/')


SCRIPT = "/gaa/home/edcastil/scripts/SVR_horario_determinista.py"
n_dimensiones = 77970
prod_train = pd.read_csv('/gaa/home/edcastil/datos/Prod_2013.csv', index_col=0)
y_train = prod_train.values

def main():
    lista_C = [10.**k for k in range (0,5)]
    lista_gamma = list(np.array([2.**k for k in range(-2, 4)])/n_dimensiones)
    lista_epsilon = list(y_train.std() * np.array([2.**k for k in range(-6, -2)]))
    
    for i in lista_C:
        for j in lista_epsilon:
            for k in lista_gamma:
                parametros = (i,j,k)
                
                program_args = (["sbatch", "-A", "gaa_serv", "-p", "gaa", "--mail-type=ALL", 
				"--mail-user=edurne.castillo@estudiante.uam.es", "--mem-per-cpu=50000", SCRIPT, str(parametros)])
                print(" ".join(program_args))
                subprocess.check_call(program_args)

if __name__ == "__main__":
	main()
