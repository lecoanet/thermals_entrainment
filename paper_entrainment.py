
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from load_data import *

import publication_settings

matplotlib.rcParams.update(publication_settings.params)
matplotlib.rcParams['ps.fonttype'] = 42

t_mar, b_mar, l_mar, r_mar = (0.07, 0.25, 0.33, 0.05)
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

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
#  return color*fraction
  return white*(1-fraction)+color*fraction

def ending_string(i):
  if i==0: return ''
  else: return '_%i' %(i+1)

sim_1e4 = DataCollection()
sim_1e3 = DataCollection()
for i in range(5):
  sim_1e3.add(dir='double_Re1e3_0p25%s' %(ending_string(i)))
  sim_1e4.add(dir='double_Re1e4_0p25_hres%s' %(ending_string(i)))

sim_1e3.average()
sim_1e4.average()

start_index_1e4 = [4,4,4,4,4]
end_index_1e4   = [198,198,198,198,198]
start_index_1e3 = [5,5,4,5,5]
end_index_1e3   = [171,198,199,166,197]

r_array_1e3 = np.concatenate( [data.r[start_index:end_index] for (data,start_index,end_index) in zip(sim_1e3.data_list,start_index_1e3,end_index_1e3)] )
r_array_1e4 = np.concatenate( [data.r[start_index:end_index] for (data,start_index,end_index) in zip(sim_1e4.data_list,start_index_1e4,end_index_1e4)] )

r_array_1e3 = np.sort(np.unique(r_array_1e3))
r_array_1e4 = np.sort(np.unique(r_array_1e4))

interp_entrainment_1e3 = []
interp_entrainment_1e4 = []
for (i,data) in enumerate(sim_1e3.data_list):
  entrainment = data.dvoldt*10**(2.5)/(data.vol*10**3)/(data.w*np.sqrt(10))
  interp_entrainment_1e3.append(np.interp(r_array_1e3,data.r[start_index_1e3[i]:end_index_1e3[i]]
                                                     ,entrainment[start_index_1e3[i]:end_index_1e3[i]]))

for (i,data) in enumerate(sim_1e4.data_list):
  entrainment = data.dvoldt*10**(2.5)/(data.vol*10**3)/(data.w*np.sqrt(10))
  interp_entrainment_1e4.append(np.interp(r_array_1e4,data.r[start_index_1e4[i]:end_index_1e4[i]]
                                                     ,entrainment[start_index_1e4[i]:end_index_1e4[i]]))

entrainment_min_1e3 = np.min(np.array(interp_entrainment_1e3),axis=0)
entrainment_max_1e3 = np.max(np.array(interp_entrainment_1e3),axis=0)
entrainment_min_1e4 = np.min(np.array(interp_entrainment_1e4),axis=0)
entrainment_max_1e4 = np.max(np.array(interp_entrainment_1e4),axis=0)

plot_axis.fill_between(r_array_1e3*10,entrainment_min_1e3,entrainment_max_1e3,
                       facecolor=change_brightness(dark_goldenrod,0.4),alpha=0.5,linewidth=0.)
plot_axis.fill_between(r_array_1e4*10,entrainment_min_1e4,entrainment_max_1e4,
                       facecolor=change_brightness( midnight_blue,0.4),alpha=0.5,linewidth=0.)

r_ave = np.linspace(0.5,4,30)
entrainment_ave_1e4 = np.zeros(len(r_ave))
r_1e4 = np.zeros(len(r_ave))
num_1e4 = np.zeros(len(r_ave))
entrainment_ave_1e3 = np.zeros(len(r_ave))
r_1e3 = np.zeros(len(r_ave))
num_1e3 = np.zeros(len(r_ave))

lw = 0.5

for (i,data) in enumerate(sim_1e3.data_list):
  index = np.argmax(data.z_ct>1.99)
  entrainment = data.dvoldt*10**(2.5)/(data.vol*10**3)/(data.w*np.sqrt(10))
  plot_axis.plot(data.r*10,entrainment,
                          color=change_brightness(dark_goldenrod,0.6),linewidth=lw)
  for j in range(1,len(r_ave)):
    mask = np.logical_and(r_ave[j-1]<data.r[:index]*10, data.r[:index]*10<r_ave[j])
    num_1e3[j] += np.sum(mask)
    r_1e3[j] += np.sum(data.r[:index][mask]*10)
    entrainment_ave_1e3[j] += np.sum(entrainment[:index][mask])

num_1e3[num_1e3==0] = 1
entrainment_ave_1e3 /= num_1e3
r_1e3 /= num_1e3

for (i,data) in enumerate(sim_1e4.data_list):
  entrainment = data.dvoldt*10**(2.5)/(data.vol*10**3)/(data.w*np.sqrt(10))
  plot_axis.plot(data.r*10,entrainment,
                          color=change_brightness(midnight_blue,0.6),linewidth=lw)
  for j in range(1,len(r_ave)):
    mask = np.logical_and(np.logical_and(r_ave[j-1]<data.r[:-1]*10, data.r[:-1]*10<r_ave[j]),data.efficiency[:-1]>0)
    num_1e4[j] += np.sum(mask)
    r_1e4[j] += np.sum(data.r[:-1][mask]*10)
    entrainment_ave_1e4[j] += np.sum(entrainment[:-1][mask])

num_1e4[num_1e4==0] = 1
entrainment_ave_1e4 /= num_1e4
r_1e4 /= num_1e4

plot_axis.plot(r_1e3[:-1],entrainment_ave_1e3[:-1],
              color=dark_goldenrod,linewidth=2,label=r'${\rm Re}=630$')

plot_axis.plot(r_1e4[:-6],entrainment_ave_1e4[:-6],
              color=midnight_blue,linewidth=2,label=r'${\rm Re}=6\,300$')

plot_axis.plot(r_ave,0.8*(r_ave)**(-1),color='k',linestyle='--',linewidth=2)
plot_axis.text(0.60,0.755,   r'$r^{-1}$',va='center',ha='center',fontsize=12,transform=plot_axis.transAxes)

plot_axis.set_xlim([0.5,4])
plot_axis.set_ylim([1e-1,1e0])
plot_axis.set_xscale('log')
plot_axis.set_yscale('log')

plot_axis.xaxis.set_minor_formatter(NullFormatter())
plot_axis.yaxis.set_minor_formatter(NullFormatter())

plot_axis.get_xaxis().set_ticks([0.5,1,2,4])
plot_axis.get_xaxis().set_ticklabels(['0.5','1','2','4'])

plot_axis.set_xlabel(r'$r_{\rm th}$',fontsize=10,fontname='Bitstream Vera Sans')
plot_axis.set_ylabel(r'$\epsilon_{\rm net}$',fontsize=10,fontname='Bitstream Vera Sans')
lg = plot_axis.legend(loc='lower left',fontsize=10)
lg.draw_frame(False)

plt.savefig('entrainment.png', dpi=600)

