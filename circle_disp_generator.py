# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:37:06 2024

@author: Admin
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from math import floor,sqrt,e,pi
from scipy import optimize
import matplotlib as mpl
import matplotlib.path as mpltPath
from matplotlib.pyplot import hist
import random
from Reflections import reflect_circle

# displacement in a circle generator

# disp_thre = 2
bin_number = 20
rand_num = 1  # do NOT change
sim_num = 10000
divide_num = 100
D = 4 # unit is um2/s
time_interval = 0.001 # unit is s, so 0.001 is 1ms
Gauss_sigma = sqrt(2*D*time_interval/divide_num*1000*1000)
# Rayleigh_sigma = 0.6
circle_diameter = 300 # unit is nm
circle_radius = circle_diameter/2 # unit is nm
switch_circle_noboundary = 1 # 0 no boundary, 1 circle
round_digit = 10000 # 10000:保留小数点后四位；100：两位
localization_precision = 30 # unit is nm
# bin_range = (0,circle_diameter+2*localization_precision)

# bin_size = disp_thre/bin_number
# bin_thre = int(floor((disp_thre*0.8)/bin_size)) 
# bins_x = np.arange(0.5*bin_size, disp_thre, bin_size, dtype=np.float32)

def twoD_diff_dis_func(x,a,b):
    
    return 2*b*x/a*np.exp(-x**2/a)

plt.close(fig='all')

x_array = np.zeros(sim_num)
y_array = np.zeros(sim_num)
x_disp_array = np.zeros(sim_num)
y_disp_array = np.zeros(sim_num)
x_start_array = np.zeros(sim_num)
y_start_array = np.zeros(sim_num)

x_individual_array = np.zeros(divide_num)
y_individual_array = np.zeros(divide_num)
x_individual_disp_array = np.zeros(divide_num)
y_individual_disp_array = np.zeros(divide_num)

counter = 0
start_time = time.time()

while counter < sim_num:
    
    x0 = random.uniform(-circle_diameter,circle_diameter)
    y0 = random.uniform(-circle_diameter,circle_diameter)
    
    if (x0**2+y0**2)<(circle_radius**2):
        
        x_start_array[counter] = x0
        y_start_array[counter] = y0
        counter_divide = 0
        while counter_divide<divide_num:
            
            x_disp = np.random.normal(0,Gauss_sigma,rand_num)
            y_disp = np.random.normal(0,Gauss_sigma,rand_num)
            x1 = x0+x_disp[0]
            y1 = y0+y_disp[0]
            
            if switch_circle_noboundary == 1:
            
                if (x1**2+y1**2)>circle_radius**2:
                    
                    x0,y0,x1,y1 = reflect_circle(x0,y0,x1,y1,circle_radius,round_digit)
                    
                while (x1**2+y1**2)>circle_radius**2:
                    
                    x0,y0,x1,y1 = reflect_circle(x0,y0,x1,y1,circle_radius,round_digit)
                
            x_individual_disp_array[counter_divide]=x1-x0
            y_individual_disp_array[counter_divide]=y1-y0
            x0 = x1
            y0 = y1
            x_individual_array[counter_divide]=x1
            y_individual_array[counter_divide]=y1
            
            
            counter_divide+=1
        
        x_array[counter] = x1
        y_array[counter] = y1
        
        x_disp_array[counter]=x1-x_start_array[counter]
        y_disp_array[counter]=y1-y_start_array[counter]
        
        counter=counter+1
    
# rotation for symmetry, not transpose
# x_array_new = np.concatenate((x_array,y_array),axis=0)
# y_array_new = np.concatenate((y_array,x_array),axis=0)

theta = np.linspace(0,2*np.pi,200)
x_circle = np.cos(theta)*circle_radius
y_circle = np.sin(theta)*circle_radius

plt.figure(figsize=(8,8))
plt.scatter(x_start_array,y_start_array,s=10)
plt.plot(x_circle,y_circle,'r--')
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(x_array,y_array,s=10)
plt.plot(x_circle,y_circle,'r--')
plt.show()

# plt.figure(figsize=(8,8))
# plt.plot(x_individual_array,y_individual_array)
# plt.plot(x_circle,y_circle,'r--')
# plt.show()

# plt.figure(figsize=(8,8))
# x_individual_hist = hist(x_individual_disp_array,bins=21,range=(-4,4),rwidth=0.8)
# y_individual_hist = hist(y_individual_disp_array,bins=21,range=(-4,4),rwidth=0.8)

# localization precision added
x_disp_loc_array = np.random.normal(0,localization_precision,sim_num)
y_disp_loc_array = np.random.normal(0,localization_precision,sim_num)
x_disp_array = x_disp_array+x_disp_loc_array
y_disp_array = y_disp_array+y_disp_loc_array

plt.figure(figsize=(8,8))
xy_disp_array = np.sqrt(x_disp_array**2+y_disp_array**2)
disp_hist = hist(xy_disp_array,bins=bin_number,range=(0,circle_diameter+2*localization_precision),rwidth=0.8)
y_results = np.array(disp_hist[0])
x_results_1 = np.array(disp_hist[1])
x_results_2 = np.roll(x_results_1,-1)
x_results = ((x_results_1+x_results_2)/2)[0:-1]

guessed_a = (np.average(x_results[0:-1]))**2
# guessed_b = (y_results[-1]/x_results[-1] + y_results[-2]/x_results[-2])/2
guessed_b = np.amax(y_results[0:-2])*sqrt(guessed_a)*e/2
        
p0 = [guessed_a,guessed_b]
        
try: 
            
    popt,_ = optimize.curve_fit(twoD_diff_dis_func,x_results,y_results,p0=p0)
    selected_drate = popt[0]/4/time_interval/1000/1000
    # selected_slope = popt[1]/np.sum(y_results)
    print('Selected Area Diffusion Rate-2D: '+'%.2f' % selected_drate)
    # print('Selected Area background-2D: '+'%.6f' % selected_slope)
    # print('see bin_x for x axis; Hist(y axis) is saved as txt')
    # save_hist = np.concatenate((x_results,y_results),axis = 0)
    # fname = file_path_Hist.replace('.npy','_Sel'+str(total_mol_select)+'.txt')
    # np.savetxt(fname,save_hist,fmt='%.5f',delimiter=' ',newline='\n')
    print('Fitted Curve:')
    print('2*%.0f*x/%.5f*exp(-x^2/%.5f)' %(popt[1],popt[0],popt[0]))
    
    smooth_x = np.linspace(0,circle_diameter+2*localization_precision,num=int(100))
    fitted_y = 2*popt[1]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
    # distribution_y = 2*popt[2]*smooth_x/popt[0]*np.exp(-smooth_x**2/popt[0])
    # background_y = popt[1]*smooth_x
    plt.plot(smooth_x, fitted_y,'k')
    # plt.plot(smooth_x, distribution_y,'r--')
    # plt.plot(smooth_x, background_y,'b--')
    plt.show()
    
except RuntimeError:
            
    print ("RuntimeError")

print("--- %.1f seconds ---" % (time.time() - start_time))


