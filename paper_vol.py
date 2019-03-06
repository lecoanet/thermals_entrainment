
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from load_data import *

import publication_settings

matplotlib.rcParams.update(publication_settings.params)
matplotlib.rcParams['ps.fonttype'] = 42

t_mar, b_mar, l_mar, r_mar = (0.05, 0.25, 0.275, 0.05)
h_plot, w_plot = (1, 1/publication_settings.golden_mean)
w_pad = 0.35

h_total = t_mar + h_plot + b_mar
w_total = l_mar + w_pad + 2*w_plot + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

plot_axes = []

for i in range(2):
  left = (l_mar+ i*(w_pad + w_plot) ) / w_total
  bottom = 1 - (t_mar + h_plot ) / h_total
  width = w_plot / w_total
  height = h_plot / h_total
  plot_axes.append(fig.add_axes([left, bottom, width, height]))

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

t = sim_1e3.data_list[0].t

white = np.array((1,1,1))
dark_goldenrod = np.array((184/255,134/255, 11/255))
midnight_blue  = np.array((25 /255, 25/255,112/255))
firebrick_red  = np.array((178/255, 34/255, 34/255))

def change_brightness(color,fraction):
  return white*(1-fraction)+color*fraction

for (i,data) in enumerate(sim_1e3.data_list):
  p_1e3_0, = plot_axes[0].plot(data.z_ct[:-2]*10,data.vol[:-2]*10**3,
                               color=change_brightness(dark_goldenrod,0.4+0.6/4*i),linewidth=2,label=r'${\rm Re}=630$')
  p_1e3_1, = plot_axes[1].plot(data.z_ct[:-2]*10,data.vol[:-2]*10**3,
                               color=change_brightness(dark_goldenrod,0.4+0.6/4*i),linewidth=2,label=r'${\rm Re}=630$')

for (i,data) in enumerate(sim_1e4.data_list):
  p_1e4_0, = plot_axes[0].plot(data.z_ct[:-2]*10,data.vol[:-2]*10**3,
                               color=change_brightness(midnight_blue,0.4+0.6/4*i),linewidth=2,label=r'${\rm Re}=6\,300$')
  p_1e4_1, = plot_axes[1].plot(data.z_ct[:-2]*10,data.vol[:-2]*10**3,
                               color=change_brightness(midnight_blue,0.4+0.6/4*i),linewidth=2,label=r'${\rm Re}=6\,300$')

plot_axes[1].text(0.3,0.7,   r'$z_{\rm ct}^3$',va='center',ha='center',fontsize=12,transform=plot_axes[1].transAxes)
plot_axes[1].plot(data.z_ct*10,(data.z_ct*10/2)**3/2.5,color='k',linestyle='--',linewidth=2)

plot_axes[0].set_xlim([0,20])
plot_axes[0].set_ylim([0,90])
plot_axes[1].set_xlim([3,20])
plot_axes[1].set_ylim([0.2,90])
plot_axes[1].set_xscale('log')
plot_axes[1].set_yscale('log')
for i in range(2):
  plot_axes[i].set_xlabel(r'$z_{\rm ct}$')
  plot_axes[i].set_ylabel(r'$V$')
plots = [p_1e3_0,p_1e4_0]
lg_0 = plot_axes[0].legend(plots,[plot.get_label() for plot in plots],loc='upper left',fontsize=10)
lg_0.draw_frame(False)
plots = [p_1e3_1,p_1e4_1]
lg_1 = plot_axes[1].legend(plots,[plot.get_label() for plot in plots],loc='lower right',fontsize=10)
lg_1.draw_frame(False)

plt.savefig('volume_both_z.png', dpi=300)
#plt.savefig('volume_both_z.eps')

