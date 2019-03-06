
import numpy as np
import matplotlib.pyplot as plt
import os.path

class SimulationData:

  def __init__(self,dir):

    self.t,self.z_ct,self.z_new,self.w,self.r,self.vol = load_data_sqrt(dir)

    if os.path.isfile("%s/mean_rho.dat" %dir):
      self.mean_rho = np.loadtxt('%s/mean_rho.dat' %dir)

    if os.path.isfile("%s/mass_below.dat" %dir):
      data = np.loadtxt("%s/mass_below.dat" %dir)
      self.mass_below = data[0,:]
      self.mass_way_below = data[1,:]
    else:
      self.mass_below = self.t*0
      self.mass_way_below = self.t*0

    self.total_mass = -4*np.pi/3*(0.05)**3

    self.calculate_dvoldt()
    self.calculate_mass()
    self.calculate_dMdt()
    self.calculate_efficiency()
    self.calculate_entrainment()

    if os.path.isfile('%s/mean_c.dat' %dir):
      data = np.loadtxt("%s/mean_c.dat" %dir)
      self.c_in = data[0,:]
      self.c_out = data[1,:]
      self.c_below = data[2,:]
      self.c_z0 = data[3,:]
      self.calculate_detrainment()

  def calculate_detrainment(self):
    c_len = len(self.c_z0)
    self.t_d = self.t[5:c_len:10]
    self.z_d = self.z_ct[5:c_len:10]
    if c_len == 172: c_len=170
    self.detrainment = (self.c_z0[9:c_len:10] - self.c_z0[:c_len:10])/(self.z_ct[9:c_len:10] - self.z_ct[:c_len:10])/(self.c_in[5:c_len:10]+self.c_out[5:c_len:10])
    
  def calculate_dvoldt(self):
    self.dvoldt = np.gradient(self.vol,self.t)

  def calculate_mass(self):
    self.mass = self.vol*self.mean_rho

  def calculate_dMdt(self):
    self.dMdt = np.zeros(len(self.t))
    i_ave = 20
    self.dMdt[:i_ave] = self.mass_below[:i_ave]/self.t[:i_ave]
    self.dMdt[i_ave:] = (self.mass_below[:-i_ave]-self.mass_below[i_ave:])/(self.t[:-i_ave]-self.t[i_ave:])

  def calculate_efficiency(self):
    self.efficiency = self.dvoldt/self.vol/self.w*self.r

  def calculate_entrainment(self):
    self.entrainment = self.dvoldt/self.vol/self.w

class DataCollection:

  def __init__(self):
    self.data_list = []

  def add(self,dir):
    self.data_list.append(SimulationData(dir))

  def average(self):
    self.t = self.data_list[0].t

    def do_average(quantity_list):
      ave = 0*getattr(self.data_list[0],quantity_list[0])
      for data in self.data_list:
        single = 0*ave+1
        for quantity in quantity_list:
          single *= getattr(data,quantity)
        ave += single
      return ave/len(self.data_list)

    self.ave_efficiency = do_average(['efficiency'])
    self.ave_entrainment = do_average(['entrainment'])
    self.ave_r = do_average(['r'])
    self.ave_w = do_average(['w'])
    self.ave_mean_rho = do_average(['mean_rho'])
    self.ave_mass = do_average(['mean_rho','vol'])
    self.ave_vol = do_average(['vol'])
    self.ave_mass_below = do_average(['mass_below'])
    self.ave_mass_way_below = do_average(['mass_way_below'])

def z_sqrt_func(t, a, t0, z0):
  return a*np.abs(t)**(1/2) + z0

def w_sqrt_func(t, a, t0, z0):
  return a/2/np.abs(t)**(1/2)

from scipy.optimize import curve_fit

def load_data_sqrt(dir):
  r_array = np.loadtxt('%s/contour_flux.dat' %dir)

  z_data = np.loadtxt('%s/cloud_top.txt' %dir)
  t = z_data[0]
  z_ct = z_data[1]

  mask = (z_ct < 1.8) & (z_ct>0.5)
  popt, pcov = curve_fit(z_sqrt_func, t[mask], z_ct[mask])
  z_new = z_sqrt_func(t,*popt)

  w = w_sqrt_func(t,*popt)
  
  dz = 2/r_array.shape[1]
  vol = np.sum(dz*np.pi*r_array**2,axis=1)
  r = np.max(r_array,axis=1)

  return (t,z_ct,z_new,w,r,vol)

