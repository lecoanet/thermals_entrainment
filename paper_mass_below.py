
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from load_data import *

import publication_settings

matplotlib.rcParams.update(publication_settings.params)
matplotlib.rcParams['ps.fonttype'] = 42

t_mar, b_mar, l_mar, r_mar = (0.05, 0.25, 0.4, 0.07)
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
dz = z_ave[1]-z_ave[0]
mass_below_ave_1e4 = np.zeros(len(z_ave))
z_1e4 = np.zeros(len(z_ave))
num_1e4 = np.zeros(len(z_ave))
mass_below_ave_1e3 = np.zeros(len(z_ave))
z_1e3 = np.zeros(len(z_ave))
num_1e3 = np.zeros(len(z_ave))

for (i,data) in enumerate(sim_1e4.data_list):
  p_1e4, = plot_axis.plot(data.z_ct[:-1]*10,data.mass_below[:-1]*10**3/(-4*np.pi/3*0.5**3),
                          color=change_brightness(midnight_blue,0.4+0.6/4*i),linewidth=1,label=r'${\rm Re}=6\,300$')
  for j in range(1,len(z_ave)):
    mask = np.logical_and(z_ave[j-1]<data.z_ct[:-1]*10, data.z_ct[:-1]*10<z_ave[j])
    num_1e4[j] += np.sum(mask)
    z_1e4[j] += np.sum(data.z_ct[:-1][mask]*10)
    mass_below_ave_1e4[j] += np.sum(data.mass_below[:-1][mask]*10**3/(-4*np.pi/3*0.5**3))

num_1e4[num_1e4==0] = 1
mass_below_ave_1e4 /= num_1e4
z_1e4 /= num_1e4

for (i,data) in enumerate(sim_1e3.data_list):
  index = np.argmax(data.z_ct>1.99)
  p_1e3, = plot_axis.plot(data.z_ct[:index]*10,data.mass_below[:index]*10**3/(-4*np.pi/3*0.5**3),
                          color=change_brightness(dark_goldenrod,0.4+0.6/4*i),linewidth=1,label=r'${\rm Re}=630$')
  for j in range(1,len(z_ave)):
    mask= np.logical_and( z_ave[j-1]<data.z_ct[:index]*10, data.z_ct[:index]*10<z_ave[j])
    num_1e3[j] += np.sum(mask)
    z_1e3[j] += np.sum(data.z_ct[:index][mask]*10)
    mass_below_ave_1e3[j] += np.sum(data.mass_below[:index][mask]*10**3/(-4*np.pi/3*0.5**3))

num_1e3[num_1e3==0] = 1
mass_below_ave_1e3 /= num_1e3
z_1e3 /= num_1e3

plot_axis.plot(z_1e4[:-1],mass_below_ave_1e4[:-1],
              color=0.7*midnight_blue,linewidth=3)

plot_axis.plot(z_1e3[:-1],mass_below_ave_1e3[:-1],
              color=0.7*dark_goldenrod,linewidth=3)


plot_axis.set_xlim([5,20])
plot_axis.set_ylim([0,0.4])
plot_axis.set_xlabel(r'$z_{\rm ct}$')
plot_axis.set_ylabel(r'$M_{\rm d}/M_0$')
plots = [p_1e3,p_1e4]
lg = plot_axis.legend(plots,[plot.get_label() for plot in plots],loc='upper left',fontsize=10)
lg.draw_frame(False)

plt.savefig('mass_below.png', dpi=300)
#plt.savefig('mass_below.eps')

