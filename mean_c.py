
import numpy as np
import h5py

r_zeros = np.loadtxt('contour_flux.dat')

(t,midpoint_x,midpoint_y) = np.loadtxt('thermal_midpoint_flux.dat')


mean_c_in_list = []
mean_c_out_list = []
mean_c_below_list = []
mean_c_z0_list = []

for i in range(20):
  f = h5py.File('dumps_dye/dumps_dye_s%i.h5' %(i+1),flag='r')
  times = len(f['scales/sim_time'])
  x = np.array(f['scales/x/1.0'])
  dx = x[1]-x[0]
  x = x.reshape((len(x),1,1))
  y = np.array(f['scales/y/1.0'])
  y = y.reshape((1,len(y),1))
  z = np.array(f['scales/z/1.0'])
  for j in range(times):
    t_index = i*10+j
    print(t_index)
    c = np.array(f['tasks/c'][j])

    r = r_zeros[t_index]
    r = r.reshape((1,1,len(r)))

    x_shift = x - midpoint_x[t_index]
    y_shift = y - midpoint_y[t_index]

    mask_in = x_shift**2 + y_shift**2 < r**2
    mask_out = x_shift**2 + y_shift**2 > r**2
    mask_below = np.copy(mask_in)
    for k in range(1,mask_below.shape[2]):
        if k < np.argmax(r[0,0,:]):
            mask_below = (x_shift**2 + y_shift**2) > r**2
        else:
            mask_below[:,:,k] = False
    num_in = np.sum(mask_in)
    num_out = np.sum(mask_out)
    num_below = np.sum(mask_below)
    if num_in > 0:
      mean_c_in = np.sum(c[mask_in]*dx**3)
    else:
      mean_c_in = 0.
    if num_below > 0:
      mean_c_below = np.sum(c[mask_below]*dx**3)
    else:
      mean_c_below = 0.
    mean_c_out = np.sum(c[mask_out]*dx**3)
    index_bot = np.argmax(r[0,0,:]>0)
    print(z[index_bot])
    mean_c_z0 = np.sum(c[:,:,:index_bot]*dx**3)
    print(mean_c_in,mean_c_out,mean_c_below,mean_c_z0)
    mean_c_in_list.append(mean_c_in)
    mean_c_out_list.append(mean_c_out)
    mean_c_below_list.append(mean_c_below)
    mean_c_z0_list.append(mean_c_z0)

mean_c_in_list = np.array(mean_c_in_list)
mean_c_out_list = np.array(mean_c_out_list)
mean_c_below_list = np.array(mean_c_below_list)
mean_c_z0_list = np.array(mean_c_z0_list)

data = np.vstack((mean_c_in_list,mean_c_out_list,mean_c_below_list,mean_c_z0_list))

np.savetxt('mean_c.dat',data)

