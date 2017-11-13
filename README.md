# PrediccionSolarEnsembles

## Interpolación mediante clear-sky
* interpolacion_punto.py -> proceso de transformación a trihorario e interpolación.
* interpolacion_cubica_punto.py -> para variables de no radiación

## Transformación de datos del centro europeo:
* netcdf_to_myp.py
* conversion_masiva.py -> realiza todo el proceso de transformación de los datos en DataMatrix 

## Experimento 01: Modelo SVR para matrices deterministas horarias
Ficheros de implementación:
* parametros_svr.py -> lanzar trabajos a la cola
* SVR_horario_determinista.py -> validación
* test_SVR -> test
* Analisis_resultados -> plots

Ficheros de resultados:
* resultados_svr_resolucion.txt -> validación
* comparaciones_svr_test.pkl -> lista con y_pred e y_test
* resultados_test_resolucion.txt -> test

## Otros:
* cambio_resolucion.py -> pasar a resolución 0.5 las matrices
* DataMatrix_NWP.py -> librería para trabajar con las matrices
* plots -> ejemplos para hacer plots

## Experimento 02: Modelo SVR para matrices deterministas trihorarias



