print('start of script')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import pickle
import os
from scipy import interpolate
from mpi4py import MPI
from dedalus import public as de
from scipy import interpolate
from scipy import optimize

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

print('done with imports %i/%i' %(rank,size))

if (rank==0) and (not os.path.isdir('data')): os.mkdir('data')

# 1. calculate cloud tops
# 2. calculate w_top
# 3. calculate horizontal axis
# 4. calculate azimuthally averaged velocities
# 5. calculate psi=0 == control volume

set = 11
index = 4

N = 512

# 1. calculate cloud tops

def cloud_top(rho,z):
  rho_ave = np.mean(np.mean(rho,axis=0),axis=0)
  rho_max = np.max(np.abs(rho_ave))
  rho_cutoff = rho_max*0.1

  for i in range(N-1,-1,-1):
    if np.abs(rho_ave[i])>rho_cutoff: break;

  return z[i]

calculate_cloud_top = True
if calculate_cloud_top:
  t_list = []
  cloud_top_list = []

  for set in range(1+rank,21,size):
    print( 'dumps_pert/dumps_pert_s%i.h5' %set )
    f = h5py.File('dumps_pert/dumps_pert_s%i.h5' %set,flag='r')
    for index in range(10):
      rho = np.array(f['tasks/rho'][index])
      z = np.array(f['scales/z/1.0'])
      t = np.array(f['scales/sim_time'][index])

      z_cloud_top = cloud_top(rho,z)
      print(t,z_cloud_top)
      t_list.append(t)
      cloud_top_list.append(z_cloud_top)
    f.close()

  t_list_list = comm.gather(t_list,root=0)
  cloud_top_list_list = comm.gather(cloud_top_list,root=0)
  if rank == 0:
    t_all = []
    cloud_top_all = []
    for (t_list,cloud_top_list) in zip(t_list_list,cloud_top_list_list):
      for (t, cloud_top) in zip(t_list,cloud_top_list):
        t_all.append(t)
        cloud_top_all.append(cloud_top)
    t = np.array(t_all)
    cloud_top = np.array(cloud_top_all)
    torder = np.argsort(t)
    t = t[torder]
    cloud_top = cloud_top[torder]
      
    data = np.array([t,cloud_top])
    np.savetxt('cloud_top.txt',data)

comm.Barrier()
data = np.loadtxt('cloud_top.txt')
(t,cloud_top) = data

# 2. calculate w_top

def z_sqrt_func(t, a, t0, z0):
  return a*np.abs(t)**(1/2) + z0

def w_sqrt_func(t, a, t0, z0):
  return a/2/np.abs(t)**(1/2)

from scipy.optimize import curve_fit

mask = (cloud_top > 0.5) & (cloud_top<1.8)
popt, pcov = curve_fit(z_sqrt_func, t[mask], cloud_top[mask])
z_fit = z_sqrt_func(t,*popt)

plt.plot(t,cloud_top)
plt.plot(t,z_fit)
plt.savefig('cloud_top.png',dpi=300)
plt.clf()

def w_top(t):
  return w_sqrt_func(t,*popt)

plt.plot(t,w_top(t))
plt.savefig('w_cloud_top.png',dpi=300)
plt.clf()

# 3. calculate horizontal axis

def midpoint(w,x,y,t0):

  index_time = list(t).index(t0)
  z_top = cloud_top[index_time]
  index_top = list(z).index(z_top)
  
  w_mask = w > 0
  w_ave = np.sum(w[w_mask])
  num = np.sum(w_mask)
  if num == 0:
    rhow = 0.
    x_rhow = 0.
    y_rhow = 0.
  else:

    xm, ym = np.meshgrid(x,y)
    xm, ym = xm.T, ym.T
    x_rhow = np.sum(xm[w_mask]*w[w_mask])/w_ave
    y_rhow = np.sum(ym[w_mask]*w[w_mask])/w_ave

    w_ave /= num

  return (w_ave,x_rhow,y_rhow)

calculate_midpoint = True
if calculate_midpoint:
  t_list = []
  midpoint_x_list = []
  midpoint_y_list = []

  for set in range(1,21):
    print( 'dumps_pert/dumps_pert_s%i.h5' %set )
    f = h5py.File('dumps_pert/dumps_pert_s%i.h5' %set,flag='r')
    x = np.array(f['scales/x/1.0'])
    y = np.array(f['scales/y/1.0'])
    z = np.array(f['scales/z/1.0'])
    for index in range(10):
      print(index)
      t0 = np.array(f['scales/sim_time'][index])
      w_ave_list = []
      x_rhow_list = []
      y_rhow_list = []
      for i in range(rank,N,size):
        w = np.array(f['tasks/w'][index,:,:,i])
        if rank == 0: print(i)

        (w_ave,x_rhow,y_rhow) = midpoint(w,x,y,t0)
        x_rhow_list.append(x_rhow)
        y_rhow_list.append(y_rhow)
        w_ave_list.append(w_ave)

      w_ave_list_list   = comm.gather(w_ave_list,root=0)
      x_rhow_list_list = comm.gather(x_rhow_list,root=0)
      y_rhow_list_list = comm.gather(y_rhow_list,root=0)

      if rank == 0:
        w_ave_all = []
        x_rhow_all = []
        y_rhow_all = []
        for (w_ave_list,x_rhow_list,y_rhow_list) in zip(w_ave_list_list,x_rhow_list_list,y_rhow_list_list):
          for (w_ave,x_rhow,y_rhow) in zip(w_ave_list,x_rhow_list,y_rhow_list):
            w_ave_all.append(w_ave)
            x_rhow_all.append(x_rhow)
            y_rhow_all.append(y_rhow)

        w_ave_all = np.array(w_ave_all)
        x_rhow_all = np.array(x_rhow_all)
        y_rhow_all = np.array(y_rhow_all)

        index_mid = np.argmax(w_ave_all)
        
        print(z[index_mid])
        print(np.max(np.abs(x_rhow_all)),np.max(np.abs(y_rhow_all)))
        print(t0,x_rhow_all[index_mid],y_rhow_all[index_mid])
        t_list.append(t0)
        np.savetxt('data/%i' %index,x_rhow_all)
        midpoint_x_list.append(x_rhow_all[index_mid])
        midpoint_y_list.append(y_rhow_all[index_mid])
    f.close()

  if rank == 0:
    t = np.array(t_list)
    midpoint_x = np.array(midpoint_x_list)
    midpoint_y = np.array(midpoint_y_list)
    data = np.array([t,midpoint_x,midpoint_y])
    np.savetxt('thermal_midpoint_flux.dat',data)

comm.Barrier()
data = np.loadtxt('thermal_midpoint_flux.dat')
(t,midpoint_x,midpoint_y) = data

# 4. calculate azimuthally averaged velocities

def average_velocities(w,x,y,z,t0):

  index_time = list(t).index(t0)
  w0 = w_top(t0)
  x0 = midpoint_x[index_time]
  y0 = midpoint_y[index_time]
  z_top = cloud_top[index_time]

  dx = x[1]-x[0]
  dy = y[1]-y[0]
  dz = z[1]-z[0]

  Nr = int(N/4)

  x=x.reshape(len(x),1)
  y=y.reshape(1,len(y))

  r = np.sqrt((x-x0)**2 + (y-y0)**2)

  w = (w-w0)

  r_bottom = np.linspace(0, dx*Nr, Nr, endpoint=False)
  r_top = r_bottom+r_bottom[1]

  w_ave = np.zeros(Nr)

  num = np.zeros(Nr)
  for i in range(Nr):
    mask = (r >= r_bottom[i]) & (r < r_top[i])
    w_ave[i] = np.sum(w[mask])
    num[i] = np.sum(mask)

  num[num == 0] = 1.
  w_ave = w_ave/num
  r = r_bottom + r_bottom[1] / 2.

  return (w_ave,r)


calculate_velocities = True
if calculate_velocities:
  t_list = []
  vz_list = []

  for set in range(1+rank,21,size):
    print( 'dumps_pert/dumps_pert_s%i.h5' %set )
    f = h5py.File('dumps_pert/dumps_pert_s%i.h5' %set,flag='r')
    for index in range(10):
      if not os.path.isfile('data/velocity_flux_s%i_%i.pkl' %(set,index)):
        print(index)
        vz = []
        for i in range(N):
          w = np.array(f['tasks/w'][index,:,:,i])
          x = np.array(f['scales/x/1.0'])
          y = np.array(f['scales/y/1.0'])
          z = np.array(f['scales/z/1.0'])
          t0 = np.array(f['scales/sim_time'][index])

          (vz_i,r) = average_velocities(w,x,y,z,t0)
          vz.append(vz_i)
        vz = np.array(vz)
        t_list.append(t0)
        print(vz.shape)
        vz_list.append(vz)
        data = {'r':r,'z':z,'vz':vz,'t':t0}
        pickle.dump(data,open("data/velocity_flux_s%i_%i.pkl" %(set,index),'wb'))
      else:
        print("data/velocity_flux_s%i_%i.pkl" %(set,index))
        data = pickle.load(open("data/velocity_flux_s%i_%i.pkl" %(set,index),'rb'))
        t0 = np.array(f['scales/sim_time'][index])
        t_list.append(t0)
        vz_list.append(data['vz'])
    f.close()

  t_list_list = comm.gather(t_list,root=0)
  vz_list_list = comm.gather(vz_list,root=0)
  if rank == 0:
    t_all = []
    vz_all = []
    for (t_list,vz_list) in zip(t_list_list,vz_list_list):
      for (t, vz) in zip(t_list,vz_list):
        t_all.append(t)
        print(vz.shape)
        vz_all.append(vz)
    t = np.array(t_all)
    vz = np.array(vz_all)
    torder = np.argsort(t)
    t = t[torder]
    vz = vz[torder]

    data = {'r':r,'z':z,'vz':vz,'t':t}
    pickle.dump(data,open("velocity_flux.pkl",'wb'))

if rank == 0:
  print('loading velocity')
  data = pickle.load(open("velocity_flux.pkl",'rb'))
  r = data['r']
  z = data['z']
  vz = data['vz']
  t = data['t']

# 5. calculate psi=0 == control volume

if rank == 0:
  r_basis_old = de.SinCos('r', int(N/4), interval=(0, np.max(r)))
  z_basis_old = de.SinCos('z', N, interval=(0, 2))
  domain_old = de.Domain([r_basis_old,z_basis_old], grid_dtype=np.float64, comm=MPI.COMM_SELF)

  r_old = domain_old.grid(0)

  w_old = domain_old.new_field()
  w_old.meta['r']['parity'] = 1
  w_old.meta['z']['parity'] = -1

  problem_old = de.LBVP(domain_old, variables=['psi'])
  problem_old.parameters['w'] = w_old

  z_basis = de.SinCos('z', N, interval=(0, 2))
  r_basis = de.Chebyshev('r', int(N/4), interval=(0, np.max(r)))
  domain = de.Domain([z_basis,r_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)

  w = domain.new_field()
  w.meta['z']['parity'] = -1

  r = domain.grid(1)

  # Poisson equation
  problem = de.LBVP(domain, variables=['psi'])
  problem.parameters['pi'] = np.pi
  problem.parameters['w'] = w
  problem.meta['psi']['z']['parity'] = -1
  problem.add_equation("dr(psi) = 2*pi*r*w")
  #problem.add_equation("dz(psi) =-2*pi*r*u")
  problem.add_bc("left(psi) = 0")

  # Build solver
  solver = problem.build_solver()
  psi = solver.state['psi']

  r_zero_list = []

  r_zero = np.zeros(N)
  r_zero_list.append(r_zero)

  for i in range(1,200):

    print(i)
    w_old['g'] = vz[i].T

    for r_i in range(int(N/4)):
      w['g'][:,r_i] = eval("interp(w,r=%f)" %r[0,r_i],problem_old.namespace).evaluate()['g'][0]

    solver.solve()

    r_zero = []
    for i_z in range(N):
      amax = np.argmax(psi['g'][i_z,:])
      print(i,i_z,amax,r[0,amax],r[0,-25])
      if amax > 15 and psi['g'][i_z,amax] > 0:
        psi_i = interpolate.interp1d(r[0,:],psi['g'][i_z])
        zero = optimize.brentq(psi_i,r[0,amax],r[0,-25])
        r_zero.append(zero)
      else:
        r_zero.append(0)

    r_zero = np.array(r_zero)
    r_zero_list.append(r_zero)

  r_zero_list = np.array(r_zero_list)

  np.savetxt('contour_flux.dat',r_zero_list)

