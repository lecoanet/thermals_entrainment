
import numpy as np
import h5py

r_zeros = np.loadtxt('contour_flux.dat')

(t,midpoint_x,midpoint_y) = np.loadtxt('thermal_midpoint_flux.dat')


mean_rho_list = []

for i in range(20):
  f = h5py.File('dumps_pert/dumps_pert_s%i.h5' %(i+1),flag='r')
  times = len(f['scales/sim_time'])
  x = np.array(f['scales/x/1.0'])
  x = x.reshape((len(x),1,1))
  y = np.array(f['scales/y/1.0'])
  y = y.reshape((1,len(y),1))
  for j in range(times):
    t_index = i*10+j
    print(t_index)
    rho = np.array(f['tasks/rho'][j])

    r = r_zeros[t_index]
    r = r.reshape((1,1,len(r)))

    print(np.max(r))

    x_shift = x - midpoint_x[t_index]
    y_shift = y - midpoint_y[t_index]

    mask = x_shift**2 + y_shift**2 < r**2
    num = np.sum(mask)
    if num > 0:
      mean_rho = np.mean(rho[mask])
    else:
      mean_rho = 0.
    print(mean_rho)
    mean_rho_list.append(mean_rho)

mean_rho_list = np.array(mean_rho_list)

np.savetxt('mean_rho.dat',mean_rho_list)

