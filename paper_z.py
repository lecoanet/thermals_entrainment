
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
#  return color*fraction
  return white*(1-fraction)+color*fraction

for (i,data) in enumerate(sim_1e3.data_list):
  p_1e3, = plot_axis.plot(data.t*np.sqrt(10),data.z_ct*10,color=change_brightness(dark_goldenrod,0.4+0.6/4*i),linewidth=2,label=r'${\rm Re}=630$')
  plot_axis.plot(data.t*np.sqrt(10),data.z_new*10,color=dark_goldenrod*0.5,linestyle='--',linewidth=0.5)

for (i,data) in enumerate(sim_1e4.data_list):
  p_1e4, = plot_axis.plot(data.t*np.sqrt(10),data.z_ct*10,color=change_brightness(midnight_blue,0.4+0.6/4*i),linewidth=2,label=r'${\rm Re}=6\,300$')
  plot_axis.plot(data.t*np.sqrt(10),data.z_new*10,color=midnight_blue*0.5,linestyle='--',linewidth=0.5)

plot_axis.set_xlim([0,60])
plot_axis.set_ylim([0,20])
plot_axis.set_xlabel(r'$t$')
plot_axis.set_ylabel(r'$z_{\rm ct}$')
plots = [p_1e3,p_1e4]
lg = plot_axis.legend(plots,[plot.get_label() for plot in plots],loc='lower right',fontsize=10)
lg.draw_frame(False)

plt.savefig('z_ct.png', dpi=300)
#plt.savefig('z_ct.eps')

