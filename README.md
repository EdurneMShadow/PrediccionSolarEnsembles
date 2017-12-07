# PrediccionSolarEnsembles

## Interpolación mediante clear-sky
* interpolacion_punto.py -> proceso de transformación a trihorario e interpolación.
* interpolacion_cubica_punto.py -> para variables de no radiación
* interpolacion_global.py -> para datasets anuales

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
* crear_matrices_trihorarias -> contiene un método para pasar de determinista horario a trihorario
* desagregar -> contiene métodos para crear índices, entre otras cosas
* obtener_clear_sky -> usa la librería pvlib para descargar datasets de clear-sky

## Experimento 02: Modelo SVR para matrices deterministas trihorarias
Ficheros de implementación:
* parametros_svr.py -> lanzar trabajos a la cola
* SVR_trihorario_validacion.py -> validación
* test_SVR_trihorario -> test
* interpolacion_global -> interpolación
* Analisis_resultados -> plots

Ficheros de resultados:
* resultados_svr_trihorario.txt -> validación
* comparaciones_svr_test_trihorario.pkl -> lista con y_pred e y_test
* resultados_test_trihorario.txt -> test
* y_interpolada_trihorario01.csv -> y_interpolada

## Experimento 03: Modelo SVR para ensembles de control
Ficheros de implementación:
* parametros_svr.py -> lanzar trabajos a la cola
* SVR_control_validacion.py -> validación
* test_SVR_control -> test
* interpolacion_global -> interpolación
* Analisis_resultados -> plots

Ficheros de resultados:
* resultados_svr_control.txt -> validación
* comparaciones_svr_test_control.pkl -> lista con y_pred e y_test
* resultados_test_control.txt -> test
