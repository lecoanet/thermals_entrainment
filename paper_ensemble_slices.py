
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
from scipy.interpolate import interp1d
from scipy.optimize import fmin
from scipy.optimize import brentq
import publication_settings
from dedalus.extras import plot_tools
import brewer2mpl
import dedalus.public as de

matplotlib.rcParams.update(publication_settings.params)

color_map = ('RdBu', 'diverging',11)
b2m = brewer2mpl.get_map(*color_map, reverse=True)
cmap1 = b2m.mpl_colormap

color_map = ('OrRd', 'sequential', 9)
b2m = brewer2mpl.get_map(*color_map, reverse=True)
b2m.colors = [[127, 0, 0],
 [179, 0, 0],
 [215, 48, 31],
 [239, 101, 72],
 [252, 141, 89],
 [253, 187, 132],
 [253, 212, 158],
 [254, 232, 200],
 [255, 255, 255]]
cmap2 = b2m.mpl_colormap
dpi = 300

t_mar, b_mar, l_mar, r_mar = (0.2, 0.4, 0.4, 0.6)
h_slice, w_slice = (2., 1.)
h_pad = 0.45
w_pad = 0.05

h_cbar, w_cbar = (h_slice,0.05)

h_total = t_mar + 1*h_pad + 2*h_slice + b_mar
w_total = l_mar + w_pad + w_cbar + 5*w_slice + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
slice_axes = []
for i in range(2):
  for j in range(5):
    left = (l_mar + j*w_slice) / w_total
    bottom = 1 - (t_mar + i*h_pad + (i+1)*h_slice ) / h_total
    width = w_slice / w_total
    height = h_slice / h_total
    slice_axes.append(fig.add_axes([left, bottom, width, height]))

# cbars
cbar_axes = []
for i in range(2):
    left = (l_mar + 5*w_slice + w_pad) / w_total
    bottom = 1 - (t_mar + (i+1)*h_slice + i*h_pad ) / h_total
    width = w_cbar / w_total
    height = h_cbar / h_total
    cbar_axes.append(fig.add_axes([left, bottom, width, height]))

# load slice data
rho_lres_list = []
rho_hres_list = []

file_num = 48
output_num = 9
contour_num = (20*file_num+output_num)//10

def ending_string(num):
  if i == 0: return ''
  else: return '_%i' %(i+1)

for i in range(5):
  ending = ending_string(i)
  f = h5py.File('double_Re1e3_0p25%s/slices_pert/slices_pert_s%i.h5' %(ending,file_num))
  rho_slice = np.array(f['tasks/rho y mid'][output_num,:,0,:])
  print(f['scales/sim_time'][output_num]/np.sqrt(0.1))
  rho_lres_list.append(rho_slice)

  x_lres = np.array(f['scales/x/1.0'])*10
  y_lres = np.array(f['scales/z/1.0'])*10

  f.close()

  f = h5py.File('double_Re1e4_0p25_hres%s/slices_pert/slices_pert_s%i.h5' %(ending,file_num))
  rho_slice = np.array(f['tasks/rho y mid'][output_num,:,0,:])
  rho_hres_list.append(rho_slice)

  x_hres = np.array(f['scales/x/1.0'])*10
  y_hres = np.array(f['scales/z/1.0'])*10

  f.close()

# load contours
r_hres_list = []
x_hres_list = []
y_hres_list = []
r_lres_list = []
x_lres_list = []
y_lres_list = []
z_hres = np.linspace(0,2,num=1024,endpoint=False)
z_lres = np.linspace(0,2,num=512,endpoint=False)
for i in range(5):
  ending = ending_string(i)

  r_zeros_lres = np.loadtxt('double_Re1e3_0p25%s/contour_flux.dat' %ending)
  (t,midpoint_x_lres,midpoint_y_lres) = np.loadtxt('double_Re1e3_0p25%s/thermal_midpoint_flux.dat' %ending)
  r_lres_list.append(r_zeros_lres)
  x_lres_list.append(midpoint_x_lres)
  y_lres_list.append(midpoint_y_lres)

  r_zeros_hres = np.loadtxt('double_Re1e4_0p25_hres%s/contour_flux.dat' %ending)
  (t,midpoint_x_hres,midpoint_y_hres) = np.loadtxt('double_Re1e4_0p25_hres%s/thermal_midpoint_flux.dat' %ending)
  r_hres_list.append(r_zeros_hres)
  x_hres_list.append(midpoint_x_hres)
  y_hres_list.append(midpoint_y_hres)


lw = 1

# plot slices
c_im = []
xm_lres, ym_lres = plot_tools.quad_mesh(x_lres,y_lres)
xm_hres, ym_hres = plot_tools.quad_mesh(x_hres,y_hres)

for i in range(2):
  for j in range(5):
    if i == 0:
      xm = xm_lres
      ym = ym_lres
      data = rho_lres_list[j]
      midpoint_x = x_lres_list[j]
      midpoint_y = y_lres_list[j]
      r_zeros = r_lres_list[j]
      z = z_lres
    elif i == 1:
      xm = xm_hres
      ym = ym_hres
      data = rho_hres_list[j]
      midpoint_x = x_hres_list[j]
      midpoint_y = y_hres_list[j]
      r_zeros = r_hres_list[j]
      z = z_hres

    c_im.append(slice_axes[5*i+j].pcolormesh(xm,ym,data.T,cmap=cmap2))

    x0 = midpoint_x[contour_num]
    y0 = midpoint_y[contour_num]
    index_bot = np.argmax(r_zeros[contour_num,:]**2-y0**2>0)
    index_top = np.argmax(r_zeros[contour_num,::-1]**2-y0**2>0)
    r_zeros[contour_num,index_bot-1] = y0
    r_zeros[contour_num,-index_top] = y0
    slice_axes[5*i+j].plot(10*(x0 + np.sqrt(r_zeros[contour_num,:]**2-y0**2)),10*z,color='k',linewidth=lw)
    slice_axes[5*i+j].plot(10*(x0 - np.sqrt(r_zeros[contour_num,:]**2-y0**2)),10*z,color='k',linewidth=lw)

for slice_axis in slice_axes:
  slice_axis.axis([-5,5,0,20])

rho_lim_list = [0.045, 0.03]

for i in range(2):
  for j in range(5):
    c_im[5*i+j].set_clim(-rho_lim_list[i],0)

# slice axis labels
for i in range(2):
  for j in range(1,5):
    plt.setp(slice_axes[5*i+j].get_yticklabels(), visible=False)

for i in range(2):
  for j in range(5):
    slice_axes[5*i+j].set_xlabel(r'$x$',labelpad=2)
    if j == 0:
      slice_axes[5*i+j].set_ylabel(r'$z$',labelpad=2)
#  if i % 4 == 1 or i % 4 == 3:
#    slice_axes[i].xaxis.set_major_locator(MaxNLocator(nbins=3))
#  else:
#    slice_axes[i].xaxis.set_major_locator(MaxNLocator(nbins=3,prune='upper'))

# colorbar
cbars = []
cbars.append(fig.colorbar(c_im[0], cax=cbar_axes[0], orientation='vertical', ticks=MaxNLocator(nbins=4)))
cbars.append(fig.colorbar(c_im[5], cax=cbar_axes[1], orientation='vertical', ticks=MaxNLocator(nbins=5)))
for i in range(2):

  cbar_axes[i].text(0.5,1.07,r'$\rho$',va='center',ha='center',fontsize=14,transform=cbar_axes[i].transAxes)  

#  if i % 2 == 0:
#    cbars.append(fig.colorbar(c_im[2*i  ], cax=cbar_axes[i], orientation='horizontal', ticks=MaxNLocator(nbins=5)))
#  else:
#    cbars.append(fig.colorbar(c_im[2*i+1], cax=cbar_axes[i], orientation='horizontal', ticks=MaxNLocator(nbins=4)))
#  cbar_axes[i].xaxis.set_ticks_position('top')
#  cbar_axes[i].xaxis.set_label_position('top')
#  cbars[i].ax.tick_params(labelsize=8)
#  if i % 2 == 0:
#    cbar_axes[i].text(0.5,7.5,r'$t=%i$' %time_list[i//2],va='center',ha='center',fontsize=10,transform=cbar_axes[i].transAxes)

plt.savefig('ensemble.png',dpi=300)


