# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:44:39 2017

@author: Edurne
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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