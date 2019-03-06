
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from load_data import *

import publication_settings

matplotlib.rcParams.update(publication_settings.params)
matplotlib.rcParams['ps.fonttype'] = 42

t_mar, b_mar, l_mar, r_mar = (0.05, 0.25, 0.4, 0.05)
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

sim_1e3 = DataCollection()
sim_1e4 = DataCollection()
for i in range(5):
  sim_1e3.add(dir='double_Re1e3_0p25%s' %(ending_string(i)))

sim_1e3.average()

sim_1e4.add(dir='double_Re1e4_0p25_hres_5')

t = sim_1e3.data_list[0].t

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

zorder_global = 1

data = sim_1e4.data_list[0]
c_len = len(data.c_z0)
for j in range(c_len//10):
  max = np.min( ((j+1)*10, c_len) )
  p_1e4, = plot_axis.plot(data.z_ct[j*10:max]*10,data.c_z0[j*10:max]/(data.c_in[j*10:max]+data.c_out[j*10:max]),
                 color=change_brightness(midnight_blue,0.4+0.6/4*i),linewidth=1,label=r'${\rm Re}=6 \ 300$',zorder=zorder_global)
  plot_axis.scatter(data.z_ct[:max:10]*10,data.c_z0[:max:10]/(data.c_in[:max:10]+data.c_out[:max:10]),
                 marker='x',s=4,color=change_brightness(midnight_blue,0.4+0.6/4*i),zorder=zorder_global)
  plot_axis.scatter(data.z_ct[9:max:10]*10,data.c_z0[9:max:10]/(data.c_in[9:max:10]+data.c_out[9:max:10]),
                 marker='o',s=4,color=change_brightness(midnight_blue,0.4+0.6/4*i),zorder=zorder_global)
zorder_global += 1

for (i,data) in enumerate(sim_1e3.data_list):
  c_len = len(data.c_z0)
  for j in range(c_len//10):
    max = np.min( ((j+1)*10, c_len) )
    p_1e3, = plot_axis.plot(data.z_ct[j*10:max]*10,data.c_z0[j*10:max]/(data.c_in[j*10:max]+data.c_out[j*10:max]),
                   color=change_brightness(dark_goldenrod,0.4+0.6/4*i),linewidth=1,label=r'${\rm Re}=630$',zorder=zorder_global)
    plot_axis.scatter(data.z_ct[:max:10]*10,data.c_z0[:max:10]/(data.c_in[:max:10]+data.c_out[:max:10]),
                   marker='x',s=4,color=change_brightness(dark_goldenrod,0.4+0.6/4*i),zorder=zorder_global)
    plot_axis.scatter(data.z_ct[9:max:10]*10,data.c_z0[9:max:10]/(data.c_in[9:max:10]+data.c_out[9:max:10]),
                   marker='o',s=4,color=change_brightness(dark_goldenrod,0.4+0.6/4*i),zorder=zorder_global)
  zorder_global += 1

plot_axis.set_xlim([5,20])
plot_axis.set_ylim([-0.001,0.05])
plot_axis.set_xlabel(r'$z_{\rm ct}$')
plot_axis.set_ylabel(r'$c_{\rm d}/c_{\rm tot}$')
plots = [p_1e3,p_1e4]
lg = plot_axis.legend(plots,[plot.get_label() for plot in plots],loc='upper right',fontsize=10)
lg.draw_frame(False)

plt.savefig('c_d.png',dpi=300)
#plt.savefig('c_d.eps')

