# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:53:39 2024

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
from math import floor,sqrt,e,pi
from scipy import optimize
import matplotlib as mpl
import matplotlib.path as mpltPath
from matplotlib.pyplot import hist

def oneD_diff_dis_func(x,a,b,c):
    
    return c*np.exp(-x**2/a)+b

def twoD_diff_dis_func(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)+b*x

def twoD_diff_dis_func_double(x,a,b,c,a2,c2):
    
    return 2*c*x/a*np.exp(-x**2/a)+2*c2*x/a2*np.exp(-x**2/a2)+b*x

def twoD_diff_dis_func_Nob(x,a,b,c):
    
    return 2*c*x/a*np.exp(-x**2/a)

disp_thre = 2
bin_number = 20
rand_num = 10000
sim_num = 100
bin_size = disp_thre/bin_number
bin_thre = int(floor((disp_thre*0.8)/bin_size)) 
bins_x = np.arange(0.5*bin_size, disp_thre, bin_size, dtype=np.float32)
Gauss_sigma = 0.4
Rayleigh_sigma = 0.6


results_array = np.zeros(sim_num)
results_array_fiterror = np.zeros(sim_num)

counter = 0
while counter < sim_num:
    
    A = np.random.rayleigh(Rayleigh_sigma, rand_num)
    averaged_histogram,_ = np.histogram(A, bins = 20, range = (0,disp_thre))
    guessed_a = (np.average(bins_x[0:bin_thre]))**2
    guessed_b = (averaged_histogram[-1]/bins_x[-1] + averaged_histogram[-2]/bins_x[-2])/2
    guessed_c = np.amax(averaged_histogram[0:bin_thre])*sqrt(guessed_a)*e/2
            
    p0 = [guessed_a,guessed_b,guessed_c]
            
    try: 
                
        popt,_ = optimize.curve_fit(twoD_diff_dis_func_Nob,bins_x,averaged_histogram,p0=p0)
        
    except RuntimeError:
                
        print ("RuntimeError")
        
    results_array[counter] = popt[0]
    counter=counter+1
    
results_array_ave = np.average(results_array)

counter = 0
while counter < sim_num:
    A = np.random.rayleigh(Rayleigh_sigma,rand_num)
    B = np.random.normal(0,Gauss_sigma,rand_num)
    averaged_histogram_fiterror,_ = np.histogram(A+B, bins = 20, range = (0,disp_thre))
    guessed_a = (np.average(bins_x[0:bin_thre]))**2
    guessed_b = (averaged_histogram[-1]/bins_x[-1] + averaged_histogram[-2]/bins_x[-2])/2
    guessed_c = np.amax(averaged_histogram[0:bin_thre])*sqrt(guessed_a)*e/2
            
    p0 = [guessed_a,guessed_b,guessed_c]
            
    try: 
                
        popt,_ = optimize.curve_fit(twoD_diff_dis_func_Nob,bins_x,averaged_histogram,p0=p0)
        
    except RuntimeError:
                
        print ("RuntimeError")
        
    results_array_fiterror[counter] = popt[0]
    counter=counter+1
    
results_array_fiterror_ave = np.average(results_array_fiterror)

A_error = (results_array_fiterror_ave-results_array_ave)/results_array_ave*100
    
# drate = popt[0] * (pixel_size/1000)**2 /4/time_interval
plt.bar(bins_x,averaged_histogram_fiterror,width=bin_size*0.8)
plt.show()





