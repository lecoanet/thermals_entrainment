
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
cmaps = [cmap1,cmap2]
dpi = 300

t_mar, b_mar, l_mar, r_mar = (0.5, 0.4, 0.45, 0.2)
h_slice, w_slice = (2., 1.)
h_pad = 0.4

h_cbar, w_cbar = (0.05, 2*w_slice)
w_pad = 0.35

h_total = t_mar + 1*h_pad + 2*h_cbar + 2*h_slice + b_mar
w_total = l_mar + 2*w_pad + 6*w_slice + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
slice_axes = []
for i in range(2):
  for j in range(6):
    left = (l_mar + (j//2)*w_pad + j*w_slice) / w_total
    bottom = 1 - (t_mar + i*h_cbar + i*h_pad + (i+1)*h_slice ) / h_total
    width = w_slice / w_total
    height = h_slice / h_total
    slice_axes.append(fig.add_axes([left, bottom, width, height]))

# cbars
cbar_axes = []
for j in range(3):
  for i in range(2):
    left = (l_mar + j*w_cbar + j*w_pad) / w_total
    bottom = 1 - (t_mar + i*h_cbar + i*h_pad + (i)*h_slice ) / h_total
    width = w_cbar / w_total
    height = h_cbar / h_total
    cbar_axes.append(fig.add_axes([left, bottom, width, height]))

# load slice data
rho_list = []
w_list = []

file_num = np.array([16,48,80])
output_num = np.array([16,9,1])
contour_num = (20*file_num+output_num)//10

for i in range(3):
  f = h5py.File('double_Re1e3_0p25_5/slices_pert/slices_pert_s%i.h5' %file_num[i])
  w_slice = np.array(f['tasks/w y mid'][output_num[i],:,0,:])
  rho_slice = np.array(f['tasks/rho y mid'][output_num[i],:,0,:])
  print(f['scales/sim_time'][output_num[i]]/np.sqrt(0.1))
  w_list.append(w_slice)
  rho_list.append(rho_slice)

  x_lres = np.array(f['scales/x/1.0'])*10
  y_lres = np.array(f['scales/z/1.0'])*10

  f.close()

  f = h5py.File('double_Re1e4_0p25_hres_5/slices_pert/slices_pert_s%i.h5' %file_num[i])
  w_slice = np.array(f['tasks/w y mid'][output_num[i],:,0,:])
  rho_slice = np.array(f['tasks/rho y mid'][output_num[i],:,0,:])
  w_list.append(w_slice)
  rho_list.append(rho_slice)

  x_hres = np.array(f['scales/x/1.0'])*10
  y_hres = np.array(f['scales/z/1.0'])*10

  f.close()

# load contours
z_hres = np.linspace(0,2,num=1024,endpoint=False)
r_zeros_hres = np.loadtxt('double_Re1e4_0p25_hres_5/contour_flux.dat')
(t,midpoint_x_hres,midpoint_y_hres) = np.loadtxt('double_Re1e4_0p25_hres_5/thermal_midpoint_flux.dat')

z_lres = np.linspace(0,2,num=512,endpoint=False)
r_zeros_lres = np.loadtxt('double_Re1e3_0p25_5/contour_flux.dat')
(t,midpoint_x_lres,midpoint_y_lres) = np.loadtxt('double_Re1e3_0p25_5/thermal_midpoint_flux.dat')

lw = 1

# plot slices
c_im = []
xm_lres, ym_lres = plot_tools.quad_mesh(x_lres,y_lres)
xm_hres, ym_hres = plot_tools.quad_mesh(x_hres,y_hres)

for i in range(6):

  if i % 2 == 0:
    xm = xm_lres
    ym = ym_lres
  elif i % 2 == 1:
    xm = xm_hres
    ym = ym_hres
  c_im.append(slice_axes[i  ].pcolormesh(xm,ym,np.sqrt(10)*w_list[i].T,cmap=cmaps[0]))
  c_im.append(slice_axes[i+6].pcolormesh(xm,ym,rho_list[i].T,cmap=cmaps[1]))

  if i % 2 == 0:
    midpoint_x = midpoint_x_lres
    midpoint_y = midpoint_y_lres
    r_zeros = r_zeros_lres
    z = z_lres
  else:
    midpoint_x = midpoint_x_hres
    midpoint_y = midpoint_y_hres
    r_zeros = r_zeros_hres
    z = z_hres
  x0 = midpoint_x[contour_num[i//2]]
  y0 = midpoint_y[contour_num[i//2]]
  index_bot = np.argmax(r_zeros[contour_num[i//2],:]**2-y0**2>0)
  index_top = np.argmax(r_zeros[contour_num[i//2],::-1]**2-y0**2>0)
  r_zeros[contour_num[i//2],index_bot-1] = y0
  r_zeros[contour_num[i//2],-index_top] = y0
  slice_axes[i  ].plot(10*(x0 + np.sqrt(r_zeros[contour_num[i//2],:]**2-y0**2)),10*z,color='k',linewidth=lw)
  slice_axes[i  ].plot(10*(x0 - np.sqrt(r_zeros[contour_num[i//2],:]**2-y0**2)),10*z,color='k',linewidth=lw)
  slice_axes[i+6].plot(10*(x0 + np.sqrt(r_zeros[contour_num[i//2],:]**2-y0**2)),10*z,color='k',linewidth=lw)
  slice_axes[i+6].plot(10*(x0 - np.sqrt(r_zeros[contour_num[i//2],:]**2-y0**2)),10*z,color='k',linewidth=lw)

for slice_axis in slice_axes:
  slice_axis.axis([-5,5,0,20])

rho_lim_list = [0.2, 0.05, 0.035]
w_lim_list = [1., 0.6, 0.5]

for i in range(3):
  c_im[4*i  ].set_clim(-w_lim_list[i],w_lim_list[i])
  c_im[4*i+2].set_clim(-w_lim_list[i],w_lim_list[i])
  c_im[4*i+1].set_clim(-rho_lim_list[i],0)
  c_im[4*i+3].set_clim(-rho_lim_list[i],0)

# slice axis labels
for i in range(1,6,2):
  plt.setp(slice_axes[  i].get_yticklabels(), visible=False)
  plt.setp(slice_axes[6+i].get_yticklabels(), visible=False)

for i in [2,4,8,10]:
  plt.setp(slice_axes[i].get_yticklabels(), visible=False)

for i in range(1,12,2):
  if i // 6 == 0:
    slice_axes[i].text(0.85,0.925,   r'$w$',va='center',ha='center',fontsize=14,transform=slice_axes[i].transAxes)
  else:
    slice_axes[i].text(0.85,0.925,r'$\rho$',va='center',ha='center',fontsize=14,transform=slice_axes[i].transAxes)

for i in [0,6]:
  slice_axes[i].text(0.05,0.918,r'${\rm Re}=630$',va='center',ha='left',fontsize=10,transform=slice_axes[i].transAxes)
for i in [1,7]:
  slice_axes[i].text(0.05,0.918,r'${\rm Re}=6\,300$',va='center',ha='left',fontsize=10,transform=slice_axes[i].transAxes)

for i in range(12):
  slice_axes[i].xaxis.tick_bottom()
  if i >= 6:
    slice_axes[i].set_xlabel(r'$x$',labelpad=2)
  if i == 0 or i == 6:
    slice_axes[i].set_ylabel(r'$z$',labelpad=2)
  if i < 6:
    plt.setp(slice_axes[i].get_xticklabels(), visible=False)
  if i % 4 == 1 or i % 4 == 3:
    slice_axes[i].xaxis.set_major_locator(MaxNLocator(nbins=3))
  else:
    slice_axes[i].xaxis.set_major_locator(MaxNLocator(nbins=3,prune='upper'))

#slice_axes.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper'))
#slice_axes[1].set_xlabel(r'$x/H$')
#slice_axes.xaxis.set_major_locator(MaxNLocator(nbins=9,prune='upper'))

# colorbar
time_list = [10,30,50]
cbars = []
for i in range(6):
  if i % 2 == 0:
    cbars.append(fig.colorbar(c_im[2*i  ], cax=cbar_axes[i], orientation='horizontal', ticks=MaxNLocator(nbins=5)))
  else:
    cbars.append(fig.colorbar(c_im[2*i+1], cax=cbar_axes[i], orientation='horizontal', ticks=MaxNLocator(nbins=4)))
  cbar_axes[i].xaxis.set_ticks_position('top')
  cbar_axes[i].xaxis.set_label_position('top')
  cbars[i].ax.tick_params(labelsize=8)
  if i % 2 == 0:
    cbar_axes[i].text(0.5,7.5,r'$t=%i$' %time_list[i//2],va='center',ha='center',fontsize=10,transform=cbar_axes[i].transAxes)

plt.savefig('thermals_hres.png',dpi=1200)


