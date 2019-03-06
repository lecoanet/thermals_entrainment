"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5

The simulation should take a few process-minutes to run.

"""

restart_num = -1
restart_file = 3

import numpy as np
from mpi4py import MPI
import time
import h5py

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly, Lz = (1, 1, 2.)
Prandtl = 1.
Reynolds = 1e4

N = 512

# Create bases and domain
x_basis = de.SinCos('x', N, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.SinCos('y', N, interval=(-Ly/2, Ly/2), dealias=3/2)
z_basis = de.SinCos('z', 2*N, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64,mesh=[64,32])

x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','rho','u','v','w','rhoz','uz','vz'])

problem.meta['p','rhoz']['x','y','z']['parity'] = 1
problem.meta['u','uz']['x']['parity'] = -1
problem.meta['u','uz']['y']['parity'] = 1
problem.meta['rho','w']['x','y']['parity'] = 1
problem.meta['rho','w']['z']['parity'] = -1
problem.meta['v','vz']['x']['parity'] = 1
problem.meta['v','vz']['y']['parity'] = -1
problem.meta['u','v']['z']['parity'] = 1
problem.meta['uz','vz']['z']['parity'] = -1

radius = 0.05
z0 = 0.15
r_width = 0.01

problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['nu'] = radius/Reynolds
problem.parameters['kappa'] = radius/(Reynolds*Prandtl)

problem.add_equation("dx(u) + dy(v) + dz(w) = 0", condition = "(nx != 0) or (ny != 0) or (nz != 0)")
problem.add_equation("p = 0", condition = "(nx == 0) and (ny == 0) and (nz == 0)")
problem.add_equation("dt(rho) - kappa*(dx(dx(rho)) + dy(dy(rho)) + dz(rhoz)) = -(u*dx(rho) + v*dy(rho) + w*rhoz)")
problem.add_equation("dt(u) - nu*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p)    = -(u*dx(u) + v*dy(u) + w*uz)")
problem.add_equation("dt(v) - nu*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p)    = -(u*dx(v) + v*dy(v) + w*vz)")
problem.add_equation("dt(w) - nu*(dx(dx(w)) + dy(dy(w)) - dz(dx(u) + dy(v))) + dz(p) + rho = -(u*dx(w) + v*dy(w) - w*dx(u) - w*dy(v))")
problem.add_equation("rhoz - dz(rho) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions
rho = solver.state['rho']
rhoz = solver.state['rhoz']
w = solver.state['w']

noise = domain.new_field()
noise.meta['x','y','z']['parity'] = 1
noise.set_scales(domain.dealias)

amp = 0.01/0.5556*0.25
cutoff = 128 # must be an even divisor of N

if restart_num == -1:

  from scipy.special import erf

  r = np.sqrt(x**2+y**2+(z-z0)**2)

  rho['g'] = ( erf( (r - radius)/r_width) - 1 )/2

  rho.differentiate('z',out=rhoz)

  cshape = domain.dist.coeff_layout.global_shape(scales=1)
  clocal = domain.dist.coeff_layout.local_shape(scales=1)
  cslices = domain.dist.coeff_layout.slices(scales=1)
  rand = np.random.RandomState(seed=4200)

  amp_noise = np.zeros(clocal)
  amp_noise_rnd = rand.standard_normal((cutoff,cutoff,cutoff))
  if np.alltrue([s.start<cutoff for s in cslices]):
    amp_noise[:,:,:cutoff] = amp_noise_rnd[cslices]

  amp_phase = np.zeros(clocal)
  amp_phase_rnd = rand.uniform(0,2*np.pi,(cutoff,cutoff,cutoff))
  if np.alltrue([s.start<cutoff for s in cslices]):
    amp_phase[:,:,:cutoff] = amp_phase_rnd[cslices]

  kx = x_basis.elements.reshape(N,1,1)[cslices[0],:,:]
  ky = y_basis.elements.reshape(1,N,1)[:,cslices[1],:]
  kz = z_basis.elements.reshape(1,1,2*N)[:,:,cslices[2]]
  k = np.sqrt(1+kz**2+kx**2+ky**2)

  logger.info('shapes:')
  logger.info(amp_phase.shape)
  logger.info(amp_noise.shape)
  logger.info(k.shape)
  logger.info(noise['c'].shape)

  noise['c'] = k**(-1/3)*amp_noise*np.sin(amp_phase)

  rho['g'] *= (1 + amp*noise['g'])

elif restart_num == 0:
  file = 'checkpoints_quarter/checkpoints_quarter_s%i.h5' %(restart_file)
else:
  file = 'checkpoints_r%i/checkpoints_r%i_s%i.h5' %(restart_num,restart_num,restart_file)

#x_basis_lres = de.SinCos('x', 256, interval=(0, Lx/2), dealias=3/2)
#y_basis_lres = de.SinCos('y', 256, interval=(0, Ly/2), dealias=3/2)
#z_basis_lres = de.Chebyshev('z', 512, interval=(0, Lz), dealias=3/2)
#domain_lres = de.Domain([x_basis_lres, y_basis_lres, z_basis_lres], grid_dtype=np.float64,mesh=[64,32])

#if restart_num > -1:
#  f = h5py.File(file,flag='r')

#  temp_field = domain_lres.new_field()

#  temp_field.require_coeff_space()
#  slices = temp_field.layout.slices((1,1,1))

#  for field in solver.state.fields:

#    temp_field.set_scales((1,1,1))
#    temp_field.meta['x']['parity'] = field.meta['x']['parity']
#    temp_field.meta['y']['parity'] = field.meta['y']['parity']
#    temp_field['c'] = np.array(f['tasks'][field.name][0][slices])
#    field.set_scales((1/2,1/2,1/2))
#    temp_field.set_scales((1,1,1))
#    field['g'] = temp_field['g']

#  solver.iteration = f['scales/iteration'][0]
#  solver.sim_time = f['scales/sim_time'][0]

#  f.close()

# Initial timestep
dt = 0.001
#dt = 0.0001 # seems to be stable

# Integration parameters
solver.stop_sim_time = 20.
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
dumps = solver.evaluator.add_file_handler('dumps_pert', sim_dt=0.1, max_writes=10)
dumps.add_task('u')
dumps.add_task('v')
dumps.add_task('w')
dumps.add_task('rho')

slices = solver.evaluator.add_file_handler('slices_pert', sim_dt=0.01, max_writes=20)
slices.add_task('interp(w, x = 0)',name='w x mid')
slices.add_task('interp(w, y = 0)',name='w y mid')
slices.add_task('interp(rho, x = 0)',name='rho x mid')
slices.add_task('interp(rho, y = 0)',name='rho y mid')

slices.add_task('interp(w, z = 1)',name='w z mid')
slices.add_task('interp(rho, z = 1)',name='rho z mid')

profiles = solver.evaluator.add_file_handler('profiles', sim_dt=0.01, max_writes=100)
profiles.add_task("integ(rho,'x','y')/Lx/Ly", name='rho ave')

checkpoints = solver.evaluator.add_file_handler('checkpoints_pert', wall_dt=60*55, max_writes=1)
checkpoints.add_system(solver.state,layout='c')

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("sqrt(u*u + v*v + w*w) / nu", name='Re')

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=0.7,
                     max_change=1.5, min_change=0.5, max_dt=0.01, threshold=0.05)
CFL.add_velocities(('u', 'v', 'w'))

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        solver.step(dt)

        if (solver.iteration-1) % 1 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
