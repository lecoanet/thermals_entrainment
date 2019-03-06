
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
import pickle

matplotlib.rcParams.update(publication_settings.params)

color_map = ('RdBu', 'diverging',11)
b2m = brewer2mpl.get_map(*color_map, reverse=True)
cmap1 = b2m.mpl_colormap

dpi = 300

t_mar, b_mar, l_mar, r_mar = (0.3, 0.35, 0.35, 0.1)
h_slice, w_slice = (2., 1.)
h_pad = 0.05
w_pad = 0.25

h_cbar, w_cbar = (0.05, w_slice)

h_total = t_mar + h_pad + h_cbar + h_slice + b_mar
w_total = l_mar + 2*w_slice + w_pad + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
slice_axes = []
for i in range(2):
    left = (l_mar + i*w_slice + i*w_pad) / w_total
    bottom = 1 - (t_mar + h_cbar + h_pad + h_slice ) / h_total
    width = w_slice / w_total
    height = h_slice / h_total
    slice_axes.append(fig.add_axes([left, bottom, width, height]))

# cbars
cbar_axes = []
for i in range(2):
    left = (l_mar + i*w_slice + i*w_pad) / w_total
    bottom = 1 - (t_mar + h_cbar ) / h_total
    width = w_cbar / w_total
    height = h_cbar / h_total
    cbar_axes.append(fig.add_axes([left, bottom, width, height]))


data_lres = pickle.load(open("double_Re1e3_0p25_5/omega.pkl",'rb'))

x_lres = data_lres['x']*10
y_lres = data_lres['z']*10

om_lres = data_lres['om']/np.sqrt(10)

data_hres = pickle.load(open("double_Re1e4_0p25_hres_5/omega.pkl",'rb'))

x_hres = data_hres['x']*10
y_hres = data_hres['z']*10

om_hres = data_hres['om']/np.sqrt(10)

#from scipy.ndimage import gaussian_filter
#om_filter = gaussian_filter(om_hres,10)

om_phi = [om_lres,om_hres]

# load contours
z_hres = np.linspace(0,2,num=1024,endpoint=False)
r_zeros_hres = np.loadtxt('double_Re1e4_0p25_hres_5/contour_flux.dat')
(t,midpoint_x_hres,midpoint_y_hres) = np.loadtxt('double_Re1e4_0p25_hres_5/thermal_midpoint_flux.dat')

z_lres = np.linspace(0,2,num=512,endpoint=False)
r_zeros_lres = np.loadtxt('double_Re1e3_0p25_5/contour_flux.dat')
(t,midpoint_x_lres,midpoint_y_lres) = np.loadtxt('double_Re1e3_0p25_5/thermal_midpoint_flux.dat')

contour_num = 95

lw = 1

# plot slices
c_im = []
xm_lres, ym_lres = plot_tools.quad_mesh(x_lres,y_lres)
xm_hres, ym_hres = plot_tools.quad_mesh(x_hres,y_hres)

for i in range(2):

  if i == 0:
    xm = xm_lres
    ym = ym_lres
  else:
    xm = xm_hres
    ym = ym_hres
  c_im.append(slice_axes[i].pcolormesh(xm,ym,om_phi[i].T,cmap=cmap1))

  if i  == 0:
    midpoint_x = midpoint_x_lres
    midpoint_y = midpoint_y_lres
    r_zeros = r_zeros_lres
    z = z_lres
  else:
    midpoint_x = midpoint_x_hres
    midpoint_y = midpoint_y_hres
    r_zeros = r_zeros_hres
    z = z_hres
  x0 = midpoint_x[contour_num]
  y0 = midpoint_y[contour_num]
  index_bot = np.argmax(r_zeros[contour_num,:]**2-y0**2>0)
  index_top = np.argmax(r_zeros[contour_num,::-1]**2-y0**2>0)
  r_zeros[contour_num,index_bot-1] = y0
  r_zeros[contour_num,-index_top] = y0
  slice_axes[i  ].plot(10*(x0 + np.sqrt(r_zeros[contour_num,:]**2-y0**2)),10*z,color='k',linewidth=lw)
  slice_axes[i  ].plot(10*(x0 - np.sqrt(r_zeros[contour_num,:]**2-y0**2)),10*z,color='k',linewidth=lw)

for slice_axis in slice_axes:
  slice_axis.axis([-5,5,0,20])

for i in range(2):
  print(np.max(om_phi[i]))

om_lim_list = [3,10]

for i in range(2):
  c_im[i].set_clim(-om_lim_list[i],om_lim_list[i])

plt.setp(slice_axes[1].get_yticklabels(), visible=False)


slice_axes[0].text(0.05,0.925,r'${\rm Re}=630$',va='center',ha='left',fontsize=10,transform=slice_axes[0].transAxes)
slice_axes[1].text(0.05,0.925,r'${\rm Re}=6\,300$',va='center',ha='left',fontsize=10,transform=slice_axes[1].transAxes)

for i in range(2):
  slice_axes[i].set_xlabel(r'$x$')

slice_axes[0].set_ylabel(r'$z$')

for i in range(2):
#  slice_axes[0].xaxis.set_major_locator(MaxNLocator(nbins=3,prune='upper'))
  slice_axes[i].xaxis.set_major_locator(MaxNLocator(nbins=3))

# colorbar
cbars = []
for i in range(2):
  cbars.append(fig.colorbar(c_im[i  ], cax=cbar_axes[i], orientation='horizontal', ticks=MaxNLocator(nbins=5)))

  cbar_axes[i].xaxis.set_ticks_position('top')
  cbar_axes[i].xaxis.set_label_position('top')
  cbars[i].ax.tick_params(labelsize=8)

  cbar_axes[i].text(0.5,5.5,r'$\omega_\phi$',va='center',ha='center',fontsize=10,transform=cbar_axes[i].transAxes)

plt.savefig('thermals_om.png',dpi=600)


