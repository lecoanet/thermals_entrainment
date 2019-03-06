
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from load_data import *

import publication_settings

matplotlib.rcParams.update(publication_settings.params)
matplotlib.rcParams['ps.fonttype'] = 42

t_mar, b_mar, l_mar, r_mar = (0.05, 0.27, 0.4, 0.07)
h_plot, w_plot = (1, 1/publication_settings.golden_mean)

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

left = (l_mar) / w_total
bottom = 1 - (t_mar + h_plot ) / h_total
width = w_plot / w_total
height = h_plot / h_total
plot_axis = fig.add_axes([left, bottom, width, height])

def ending_string(i):
  if i==0: return ''
  else: return '_%i' %(i+1)

sim_1e4 = DataCollection()
sim_1e3 = DataCollection()
for i in range(5):
  sim_1e4.add(dir='double_Re1e4_0p25_hres%s' %(ending_string(i)))
  sim_1e3.add(dir='double_Re1e3_0p25%s' %(ending_string(i)))

sim_1e4.average()
sim_1e3.average()

t = sim_1e4.data_list[0].t

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

z_ave = np.linspace(0,20,40)
mass_below_ave_1e4 = np.zeros(len(z_ave))
mass_1e4 = np.zeros(len(z_ave))
mean_rho_1e4 = np.zeros(len(z_ave))
entrainment_ave_1e4 = np.zeros(len(z_ave))
detrainment_ave_1e4 = np.zeros(len(z_ave))
z_1e4 = np.zeros(len(z_ave))
num_1e4 = np.zeros(len(z_ave))
entrainment_ave_1e3 = np.zeros(len(z_ave))
detrainment_ave_1e3 = np.zeros(len(z_ave))
z_1e3 = np.zeros(len(z_ave))
num_1e3 = np.zeros(len(z_ave))

for (i,data) in enumerate(sim_1e4.data_list):
  entrainment = data.dvoldt*10**(2.5)/(data.vol*10**3)/(data.w*np.sqrt(10))
  detrainment = np.gradient(data.mass_below,data.z_ct*10)/data.mass
  for j in range(1,len(z_ave)):
    mask = np.logical_and(np.logical_and(z_ave[j-1]<data.z_ct[:-1]*10, data.z_ct[:-1]*10<z_ave[j]),data.efficiency[:-1]>0)
    num_1e4[j] += np.sum(mask)
    z_1e4[j] += np.sum(data.z_ct[:-1][mask]*10)
    mass_below_ave_1e4[j] += np.sum(data.mass_below[:-1][mask])
    entrainment_ave_1e4[j] += np.sum(entrainment[:-1][mask])
    detrainment_ave_1e4[j] += np.sum(detrainment[:-1][mask])
    mass_1e4[j] += np.sum(data.mass[:-1][mask])
    mean_rho_1e4[j] += np.sum(data.mean_rho[:-1][mask])

num_1e4[num_1e4==0] = 1
mass_below_ave_1e4 /= num_1e4
mass_1e4 /= num_1e4
mean_rho_1e4 /= num_1e4
entrainment_ave_1e4 /= num_1e4
detrainment_ave_1e4 /= num_1e4
z_1e4 /= num_1e4

for (i,data) in enumerate(sim_1e3.data_list):
  index = np.argmax(data.z_ct>1.99)
  entrainment = data.dvoldt*10**(2.5)/(data.vol*10**3)/(data.w*np.sqrt(10))
  for j in range(1,len(z_ave)):
    mask= np.logical_and(np.logical_and(np.logical_and( z_ave[j-1]<data.z_ct[:index]*10, data.z_ct[:index]*10<z_ave[j])
                                                                                       , ~np.isnan(entrainment[:index]))
                                                                                       , ~np.isinf(entrainment[:index]))
    num_1e3[j] += np.sum(mask)
    z_1e3[j] += np.sum(data.z_ct[:index][mask]*10)
    entrainment_ave_1e3[j] += np.sum(entrainment[:index][mask])

entrainment_ave_1e3[entrainment_ave_1e3==np.nan] = 0.

num_1e3[num_1e3==0] = 1
entrainment_ave_1e3 /= num_1e3
z_1e3 /= num_1e3

z_dye = np.linspace(0,20,20)
z_1e3_dye = np.zeros(len(z_dye))
detrainment_ave_1e3 = np.zeros(len(z_dye))
num_1e3_dye = np.zeros(len(z_dye))
for (i,data) in enumerate(sim_1e3.data_list):
  index = np.argmax(data.z_d>1.99)
  for j in range(1,len(z_dye)):
    mask= np.logical_and( z_dye[j-1]<data.z_d[:index]*10, data.z_d[:index]*10<z_dye[j])
    num_1e3_dye[j] += np.sum(mask)
    z_1e3_dye[j] += np.sum(data.z_d[:index][mask]*10)
    detrainment_ave_1e3[j] += np.sum(data.detrainment[:index][mask]/10)

num_1e3_dye[num_1e3_dye==0] = 1
detrainment_ave_1e3 /= num_1e3_dye
z_1e3_dye /= num_1e3_dye

plot_axis.plot(z_1e3,entrainment_ave_1e3,color=dark_goldenrod,linewidth=2,label=r'${\rm Re}=630$')
plot_axis.plot(z_1e4,entrainment_ave_1e4,color=midnight_blue,linewidth=2,label=r'${\rm Re}=6\,300$')

plot_axis.plot(z_1e3_dye,(z_1e3_dye/10)**(-1)*0.012,color='k',linestyle='--')

plot_axis.plot(z_1e3_dye,detrainment_ave_1e3,linewidth=2,color=dark_goldenrod,linestyle=':')
plot_axis.plot(z_1e4,detrainment_ave_1e4,linewidth=2,color=midnight_blue,linestyle=':')

plot_axis.text(0.3 ,0.86,   r'$\epsilon_{\rm net}$',va='center',ha='center',fontsize=12,transform=plot_axis.transAxes)
plot_axis.text(0.43,0.53,   r'$\delta$',va='center',ha='center',fontsize=12,transform=plot_axis.transAxes)

start_1e3 = np.argmax(z_1e3 > 6)
end_1e3 = np.argmax(z_1e3 > 16)
start_1e4 = np.argmax(z_1e4 > 6)
end_1e4 = np.argmax(z_1e4 > 16)

mask_1e3 = detrainment_ave_1e3[start_1e3:end_1e3] > 0
mask_1e4 = detrainment_ave_1e4[start_1e4:end_1e4] > 0

print(np.mean(entrainment_ave_1e3[start_1e3:end_1e3]))
print(np.mean(entrainment_ave_1e4[start_1e4:end_1e4]))
print(np.mean(detrainment_ave_1e3[start_1e3:end_1e3][mask_1e3]))
print(np.mean(detrainment_ave_1e4[start_1e4:end_1e4][mask_1e4]))

plot_axis.set_yscale('log')
plot_axis.set_xscale('log')
plot_axis.set_xlim([5,20])
plot_axis.set_ylim([1e-3,2])
plot_axis.set_xlabel(r'$z_{\rm ct}$')
plot_axis.set_ylabel(r'$\epsilon_{\rm net}, \, \delta$')
lg = plot_axis.legend(loc='upper right',fontsize=10)
lg.draw_frame(False)

plt.savefig('detrainment.png', dpi=300)
#plt.savefig('detrainment.eps')

