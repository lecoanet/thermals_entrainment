
import numpy as np
import h5py

r_zeros = np.loadtxt('contour_flux.dat')

(t,midpoint_x,midpoint_y) = np.loadtxt('thermal_midpoint_flux.dat')

mass_list = []
mass2_list = []

for i in range(20):
  f = h5py.File('dumps_pert/dumps_pert_s%i.h5' %(i+1),flag='r')
  times = len(f['scales/sim_time'])
  x = np.array(f['scales/x/1.0'])
  y = np.array(f['scales/y/1.0'])
  z = np.array(f['scales/z/1.0'])
  dx = x[1]-x[0]
  dy = y[1]-y[0]
  dz = z[1]-z[0]
  dV = dx*dy*dz
  for j in range(times):
    t_index = i*10+j
    print(t_index)
    rho = np.array(f['tasks/rho'][j])

    r = r_zeros[t_index]

    print(np.max(r))

    index_bot = np.argmax(r>0)
    print(z[index_bot])
    mass_below = np.sum(rho[:,:,:index_bot]*dV)

    shift = (np.max(r)/4)//dz
    index_shift = index_bot - shift
    if index_shift < 0: index_shift = 0
    print(z[index_shift])
    mass_far_below = np.sum(rho[:,:,:index_shift]*dV)

    print(mass_below)
    print(mass_far_below)
    mass_list.append(mass_below)
    mass2_list.append(mass_far_below)

mass_list = np.array(mass_list)
mass2_list = np.array(mass2_list)

data = np.vstack((mass_list,mass2_list))

np.savetxt('mass_below.dat',data)

