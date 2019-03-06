
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from load_data import *

import publication_settings

matplotlib.rcParams.update(publication_settings.params)
matplotlib.rcParams['ps.fonttype'] = 42

t_mar, b_mar, l_mar, r_mar = (0.05, 0.225, 0.275, 0.05)
h_plot, w_plot = (1, 1/publication_settings.golden_mean)

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_plot + r_mar

width = 3.45
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
  sim_1e3.add(dir='double_Re1e3_0p25%s' %(ending_string(i)))
  sim_1e4.add(dir='double_Re1e4_0p25_hres%s' %(ending_string(i)))

sim_1e3.average()
sim_1e4.average()

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

start_index_1e4 = [4,4,4,4,4]
end_index_1e4   = [198,198,198,198,198]
start_index_1e3 = [5,5,4,5,5]
end_index_1e3   = [171,198,199,166,197]

z_array_1e3 = np.concatenate( [data.z_ct[start_index:end_index] for (data,start_index,end_index) in zip(sim_1e3.data_list,start_index_1e3,end_index_1e3)] )
z_array_1e4 = np.concatenate( [data.z_ct[start_index:end_index] for (data,start_index,end_index) in zip(sim_1e4.data_list,start_index_1e4,end_index_1e4)] )

z_array_1e3 = np.sort(np.unique(z_array_1e3))
z_array_1e4 = np.sort(np.unique(z_array_1e4))

interp_efficiency_1e3 = []
interp_efficiency_1e4 = []
for (i,data) in enumerate(sim_1e3.data_list):
  interp_efficiency_1e3.append(np.interp(z_array_1e3,data.z_ct[start_index_1e3[i]:end_index_1e3[i]]
                                                    ,data.efficiency[start_index_1e3[i]:end_index_1e3[i]]))

for (i,data) in enumerate(sim_1e4.data_list):
  interp = np.interp(z_array_1e4,data.z_ct[start_index_1e4[i]:end_index_1e4[i]]
                                                    ,data.efficiency[start_index_1e4[i]:end_index_1e4[i]])
  mask = (z_array_1e4 > data.z_ct[end_index_1e4[i]-1])
  interp[mask] = -100
  interp_efficiency_1e4.append(interp)

efficiency_min_1e3 = np.min(np.array(interp_efficiency_1e3),axis=0)
efficiency_max_1e3 = np.max(np.array(interp_efficiency_1e3),axis=0)
efficiency_min_1e4 = np.min(np.abs(np.array(interp_efficiency_1e4)),axis=0)
efficiency_max_1e4 = np.max(np.array(interp_efficiency_1e4),axis=0)

plot_axis.fill_between(z_array_1e3*10,efficiency_min_1e3,efficiency_max_1e3,
                       facecolor=change_brightness(dark_goldenrod,0.4),alpha=0.5,linewidth=0.)
plot_axis.fill_between(z_array_1e4*10,efficiency_min_1e4,efficiency_max_1e4,
                       facecolor=change_brightness( midnight_blue,0.4),alpha=0.5,linewidth=0.)

z_ave = np.linspace(0,20,40)
efficiency_ave_1e4 = np.zeros(len(z_ave))
z_1e4 = np.zeros(len(z_ave))
num_1e4 = np.zeros(len(z_ave))
efficiency_ave_1e3 = np.zeros(len(z_ave))
z_1e3 = np.zeros(len(z_ave))
num_1e3 = np.zeros(len(z_ave))


for (i,data) in enumerate(sim_1e3.data_list):
  plot_axis.plot(data.z_ct*10,data.efficiency,color=change_brightness(dark_goldenrod,0.6),linewidth=0.5)
  index = np.argmax(data.z_ct>1.99)
  for j in range(1,len(z_ave)):
    mask = np.logical_and(z_ave[j-1]<data.z_ct[:index]*10, data.z_ct[:index]*10<z_ave[j])
    num_1e3[j] += np.sum(mask)
    z_1e3[j] += np.sum(data.z_ct[:index][mask]*10)
    efficiency_ave_1e3[j] += np.sum(data.efficiency[:index][mask])

num_1e3[num_1e3==0] = 1
efficiency_ave_1e3 /= num_1e3
z_1e3 /= num_1e3

for (i,data) in enumerate(sim_1e4.data_list):
  lw = 0.5
  plot_axis.plot(data.z_ct*10,data.efficiency,color=change_brightness(midnight_blue,0.6),linewidth=lw)
  for j in range(1,len(z_ave)):
    mask = np.logical_and(np.logical_and(z_ave[j-1]<data.z_ct[:-1]*10, data.z_ct[:-1]*10<z_ave[j]),data.efficiency[:-1]>0)
    num_1e4[j] += np.sum(mask)
    z_1e4[j] += np.sum(data.z_ct[:-1][mask]*10)
    efficiency_ave_1e4[j] += np.sum(data.efficiency[:-1][mask])

num_1e4[num_1e4==0] = 1
efficiency_ave_1e4 /= num_1e4
z_1e4 /= num_1e4

p_1e3, = plot_axis.plot(z_1e3,efficiency_ave_1e3,color=dark_goldenrod,linewidth=2,label=r'${\rm Re}=630$')
p_1e4, = plot_axis.plot(z_1e4,efficiency_ave_1e4,color=midnight_blue ,linewidth=2,label=r'${\rm Re}=6\,300$')

start_1e3 = np.argmax(z_1e3 > 6)
end_1e3 = np.argmax(z_1e3 > 16)
start_1e4 = np.argmax(z_1e4 > 6)
end_1e4 = np.argmax(z_1e4 > 16)

print(np.mean(efficiency_ave_1e3[start_1e3:end_1e3]))
print(np.mean(efficiency_ave_1e4[start_1e4:end_1e4]))
print(3/np.mean(efficiency_ave_1e3[start_1e3:end_1e3]))
print(3/np.mean(efficiency_ave_1e4[start_1e4:end_1e4]))

plot_axis.set_xlim([5,20])
plot_axis.set_ylim([0,1])
plot_axis.set_xlabel(r'$z_{\rm ct}$')
plot_axis.set_ylabel(r'$e$')
plots = [p_1e3,p_1e4]
lg = plot_axis.legend(plots,[plot.get_label() for plot in plots],loc='upper right',fontsize=10)
lg.draw_frame(False)

plt.savefig('efficiency.png', dpi=600)
#plt.savefig('efficiency.eps')

