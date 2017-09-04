# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:44:39 2017

@author: Edurne
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import desagregar as lib


r_original = pd.read_csv('./Plots/radiacion_original.csv', index_col=0)
r_interpolado = pd.read_csv('./Plots/radiacion_interpolada.csv', index_col=0)

dia_original = r_original[:24]
dia_interpolado = r_interpolado[:24]

y = np.arange(0,24,1)
x1 = list(dia_original[r_original.columns[0]])
x2 = list(dia_original[r_original.columns[1]])
x3 = list(dia_original[r_original.columns[2]])
x4 = list(dia_original[r_original.columns[3]])
x5 = list(dia_original[r_original.columns[4]])

z1 = list(dia_interpolado[r_interpolado.columns[0]])
z2 = list(dia_interpolado[r_interpolado.columns[1]])
z3 = list(dia_interpolado[r_interpolado.columns[2]])
z4 = list(dia_interpolado[r_interpolado.columns[3]])
z5 = list(dia_interpolado[r_interpolado.columns[4]])


plt.figure( figsize = (15, 18) )
plt.subplot(3, 2, 1)
plt.title(dia_original.columns[0])
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y,x1, label = 'original')
_ = plt.plot(y, z1, label = 'interpolado')
plt.legend(loc = 'best')

plt.subplot(3, 2, 2)
plt.title(dia_original.columns[1])
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y, x2, label = 'original')
_ = plt.plot(y, z2, label = 'interpolado')
plt.legend(loc = 'best')

plt.subplot(3, 2, 3)
plt.title(dia_original.columns[2])
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y, x3, label = 'original')
_ = plt.plot(y, z3, label = 'interpolado')
plt.legend(loc = 'best')

plt.subplot(3, 2, 4)
plt.title(dia_original.columns[3])
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y, x4, label = 'original')
_ = plt.plot(y, z4, label = 'interpolado')
plt.legend(loc = 'best')

plt.subplot(3, 2, 5)
plt.title(dia_original.columns[4])
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y, x5, label = 'original')
_ = plt.plot(y, z5, label = 'interpolado')
plt.legend(loc = 'best')

plt.tight_layout()
plt.savefig('originalVSinterpolado.pdf')
plt.show()

#=========================CLEAR-SKY=============================
cs_horario = pd.read_csv('./Plots/cs_horario.csv', index_col=0)
dia_cs = cs_horario[:24]

cs_trihorario = pd.read_csv('./Plots/cs_trihorario.csv', index_col=0)
dia_cs_3h = cs_trihorario[:8]

y = np.arange(0,24,1)
y1 = np.arange(0,24,3)
x1 = list(dia_cs[dia_cs.columns[0]])
x2 = list(dia_cs_3h[dia_cs_3h.columns[0]])

plt.figure( figsize = (15, 5) )
plt.subplot(1, 2, 1)
plt.title('cs_1h')
plt.xlabel('Horas')
plt.ylabel('Clear-Sky')
plt.xticks(y)
_ = plt.plot(y, x1)

plt.subplot(1, 2, 2)
plt.title('cs_3h_acc')
plt.xlabel('Horas')
plt.ylabel('Clear-Sky')
plt.xticks(y1)
_ = plt.plot(y1, x2)

plt.tight_layout()
plt.savefig('clear_sky.pdf')
plt.show()

#=============================================
r_trihoraria = pd.read_csv('./Plots/radiacion_trihoraria.csv', index_col=0)
dia_r_3h = r_trihoraria[:8]

t1 = list(dia_r_3h[r_trihoraria.columns[0]])
t2 = list(dia_r_3h[r_trihoraria.columns[1]])
t3 = list(dia_r_3h[r_trihoraria.columns[2]])
t4 = list(dia_r_3h[r_trihoraria.columns[3]])
t5 = list(dia_r_3h[r_trihoraria.columns[4]])

plt.figure( figsize = (15, 20) )
plt.subplot(5, 2, 1)
plt.title('FDIR_1h')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y, x1)

plt.subplot(5, 2, 2)
plt.title('FDIR_3h')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y1)
_ = plt.plot(y1, t1, '#FFA500')


plt.subplot(5, 2, 3)
plt.title('CDIR_1h')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y, x2)

plt.subplot(5, 2, 4)
plt.title('CDIR_3h_acc')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y1)
_ = plt.plot(y1, t2, '#FFA500')


plt.subplot(5, 2, 5)
plt.title('SSR_1h')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y, x3)

plt.subplot(5, 2, 6)
plt.title('SSR_3h_acc')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y1)
_ = plt.plot(y1, t3, '#FFA500')


plt.subplot(5, 2, 7)
plt.title('SSRC_1h')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y, x4)

plt.subplot(5, 2, 8)
plt.title('SSRC_3h_acc')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y1)
_ = plt.plot(y1, t4, '#FFA500')


plt.subplot(5, 2, 9)
plt.title('SSRD_1h')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y, x5)

plt.subplot(5, 2, 10)
plt.title('SSRD_3h_acc')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y1)
_ = plt.plot(y1, t5,'#FFA500')


plt.tight_layout()
plt.savefig('rad_horariaVStrihoraria.pdf')
plt.show()

#===================MAE DIARIO=========================
def plot_mae_diario(r_original, r_interpolado, maximo_error, max_error):
    indice = lib.obtener_dia_completo(maximo_error)
    indice2 = lib.obtener_dia_completo(max_error)
    fdir_original_error = r_original['(-0.125, 38.625) FDIR'].loc[indice]
    fdir_interpolado_error = r_interpolado['(-0.125, 38.625) FDIR'].loc[indice]
    
        
    cdir_original_error = r_original['(-0.125, 38.625) CDIR'].loc[indice2]
    cdir_interpolado_error = r_interpolado['(-0.125, 38.625) CDIR'].loc[indice2]
    
    ssr_original_error = r_original['(-0.125, 38.625) SSR'].loc[indice2]
    ssr_interpolado_error = r_interpolado['(-0.125, 38.625) SSR'].loc[indice2]
    
    ssrc_original_error = r_original['(-0.125, 38.625) SSRC'].loc[indice2]
    ssrc_interpolado_error = r_interpolado['(-0.125, 38.625) SSRC'].loc[indice2]
    
    ssrd_original_error = r_original['(-0.125, 38.625) SSRD'].loc[indice2]
    ssrd_interpolado_error = r_interpolado['(-0.125, 38.625) SSRD'].loc[indice2]
    
    y = np.arange(0,24,1)
    x = np.arange(0, 3000001, 500000)
    
    plt.figure( figsize = (15, 20) )
    plt.subplot(5, 2, 1)
    plt.title('FDIR interpolado vs original')
    plt.xlabel('Horas')
    plt.ylabel('Radiación')
    plt.xticks(y)
    plt.yticks(x)
    _ = plt.plot(y,list(fdir_original_error), label = 'original')
    _ = plt.plot(y, list(fdir_interpolado_error), label = 'interpolado')
    plt.legend(loc = 'best')
    
    plt.subplot(3, 2, 2)
    plt.title('CDIR interpolado vs original')
    plt.xlabel('Horas')
    plt.ylabel('Radiación')
    plt.xticks(y)
    plt.yticks(x)
    _ = plt.plot(y,list(cdir_original_error), label = 'original')
    _ = plt.plot(y, list(cdir_interpolado_error), label = 'interpolado')
    plt.legend(loc = 'best')
    
    plt.subplot(3, 2, 3)
    plt.title('SSR interpolado vs original')
    plt.xlabel('Horas')
    plt.ylabel('Radiación')
    plt.xticks(y)
    plt.yticks(x)
    _ = plt.plot(y,list(ssr_original_error), label = 'original')
    _ = plt.plot(y, list(ssr_interpolado_error), label = 'interpolado')
    plt.legend(loc = 'best')
    
    plt.subplot(3, 2, 4)
    plt.title('SSRC interpolado vs original')
    plt.xlabel('Horas')
    plt.ylabel('Radiación')
    plt.xticks(y)
    plt.yticks(x)
    _ = plt.plot(y,list(ssrc_original_error), label = 'original')
    _ = plt.plot(y, list(ssrc_interpolado_error), label = 'interpolado')
    plt.legend(loc = 'best')
    
    plt.subplot(3, 2, 5)
    plt.title('SSRD interpolado vs original')
    plt.xlabel('Horas')
    plt.ylabel('Radiación')
    plt.xticks(y)
    plt.yticks(x)
    _ = plt.plot(y,list(ssrd_original_error), label = 'original')
    _ = plt.plot(y, list(ssrd_interpolado_error), label = 'interpolado')
    plt.legend(loc = 'best')
    
    plt.savefig('errorMae.pdf')
    plt.show()


#===========================INTERPOLACION A MANO==============================
fdir = [0,0,0,0,0,0,0,444.985162,15937.29782,28106.37257,49371.75912,58847.84324,63784.07441,20279.65543,18718.84973,15715.04536,109712.7838,62398.43392,17584.26585,0,0,0,0,0]
fdir = np.array(fdir)
r_original = pd.read_csv('./Plots/radiacion_original.csv', index_col=0)
indice  = lib.obtener_dia_completo(2015032200)
fdir_original =  r_original['(-0.125, 38.625) FDIR'].loc[indice]

cdir = [0,0,0,0,0,0,0,181396.7204,650389.9228,1147001.311,1895593.414,2259420.893,2448943.962,3257962.521,3007216.324,2524649.838,5158611.67,2933926.117,826798.8099,0,0,0,0,0]
cdir = np.array(cdir)
cdir_original =  r_original['(-0.125, 38.625) CDIR'].loc[indice]

ssr = [0,0,0,0,0,0,0,32552.9214,116717.0608,205837.4784,400262.9481,477086.7322,517105.3678,572512.1589,528449.2682,443649.2807,1088024.837,618807.0668,174387.6711,0,0,0,0,0]
ssr = np.array(ssr)
ssr_original =  r_original['(-0.125, 38.625) SSR'].loc[indice]

ssrc = [0,0,0,0,0,0,0,193230.505,692819.4343,1221828.278,1800320.146,2145861.513,2325859.078,3174213.557,2929913.025,2459751.36,5050573.102,2872480.682,809482.8798,0,0,0,0,0]
ssrc = np.array(ssrc)
ssrc_original =  r_original['(-0.125, 38.625) SSRC'].loc[indice]

ssrd = [0,0,0,0,0,0,0,40942.43031,146797.2741,258885.7236,496250.1958,591497.1281,641112.6518,718270.6751,662989.6094,556600.0012,1355057.969,770680.4281,217182.5266,0,0,0,0,0]
ssrd = np.array(ssr)
ssrd_original =  r_original['(-0.125, 38.625) SSRD'].loc[indice]

y = np.arange(0,24,1)

plt.figure( figsize = (15, 20) )
plt.subplot(5, 2, 1)
plt.title('FDIR interpolado vs original')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y,fdir_original, label = 'original')
_ = plt.plot(y, fdir, label = 'interpolado')
plt.legend(loc = 'best')

plt.subplot(3, 2, 2)
plt.title('CDIR interpolado vs original')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y,cdir_original, label = 'original')
_ = plt.plot(y, cdir, label = 'interpolado')
plt.legend(loc = 'best')

plt.subplot(3, 2, 3)
plt.title('SSR interpolado vs original')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y,ssr_original, label = 'original')
_ = plt.plot(y, ssr, label = 'interpolado')
plt.legend(loc = 'best')

plt.subplot(3, 2, 4)
plt.title('SSRC interpolado vs original')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y,ssrc_original, label = 'original')
_ = plt.plot(y, ssrc, label = 'interpolado')
plt.legend(loc = 'best')

plt.subplot(3, 2, 5)
plt.title('SSRD interpolado vs original')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y,ssrd_original, label = 'original')
_ = plt.plot(y, ssrd, label = 'interpolado')
plt.legend(loc = 'best')

plt.savefig('InterpolacionAMano.pdf')
plt.show()
#============================================================================
cs = pd.read_csv('./Plots/cs_horario.csv', index_col=0)
dia_cs = cs.loc[indice]
dia_cs = list(dia_cs['(-0.125, 38.625) CS H'])
plt.title('CS vs original')
plt.xlabel('Horas')
plt.ylabel('Radiación')
plt.xticks(y)
_ = plt.plot(y,fdir_original, label = 'original')
_ = plt.plot(y, dia_cs, label = 'clear-sky')
plt.legend(loc = 'best')













