# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 09:32:12 2024

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
from sphere_disp_generator_func import sphere_disp_generator_func
from circle_disp_generator_func import circle_disp_generator_func

Dtrue = 4 # unit is um2/s
bin_number = 30
localization_precision = 30 # unit is nm
sim_num = 100
file_path=r"E:\BaiduSyncdisk\Python_programs\sphere-circle-diffusion\Hist.npy"

dia_num = 2
diameter_array = np.linspace(100,650,num=dia_num)
drate_circle=np.ones_like(diameter_array)
drate_sphere=np.ones_like(diameter_array)
hist_x_circle = np.zeros((dia_num,bin_number))
hist_y_circle = np.zeros((dia_num,bin_number))
hist_x_sphere = np.zeros((dia_num,bin_number))
hist_y_sphere = np.zeros((dia_num,bin_number))

    # bin_range = circle_diameter+2*localization_precision
    # if bin_range > 300:
    #     bin_range = 300

counter = 0
while counter<dia_num:
    
    hist_x_circle[counter,:],hist_y_circle[counter,:],drate_circle[counter] = circle_disp_generator_func(diameter_array[counter],
                                                       Dtrue,bin_number,
                                                       localization_precision,
                                                       sim_num)
    
    hist_x_sphere[counter,:],hist_y_sphere[counter,:],drate_sphere[counter] = sphere_disp_generator_func(diameter_array[counter],
                                                       Dtrue,bin_number,
                                                       localization_precision,
                                                       sim_num)
    counter+=1
    print(str(counter))
    
plt.figure(figsize=(8,8))
plt.plot(diameter_array,drate_circle,'k')
plt.plot(diameter_array,drate_sphere,'r')

save_hist = np.stack((diameter_array,drate_circle,drate_sphere),axis = 1)

fname = file_path.replace('.npy','-D%.3f-bn%.0f-lo%.0f-sn%.0f.txt' %(Dtrue,bin_number,localization_precision,sim_num))
np.savetxt(fname,save_hist,fmt='%.4f',delimiter=' ',newline='\n')

fname = file_path.replace('.npy','-D%.3f-bn%.0f-lo%.0f-sn%.0f-circlehist-x.txt' %(Dtrue,bin_number,localization_precision,sim_num))
np.savetxt(fname,np.transpose(hist_x_circle),fmt='%.2f',delimiter=' ',newline='\n')

fname = file_path.replace('.npy','-D%.3f-bn%.0f-lo%.0f-sn%.0f-circlehist-y.txt' %(Dtrue,bin_number,localization_precision,sim_num))
np.savetxt(fname,np.transpose(hist_y_circle),fmt='%.2f',delimiter=' ',newline='\n')

fname = file_path.replace('.npy','-D%.3f-bn%.0f-lo%.0f-sn%.0f-spherehist-x.txt' %(Dtrue,bin_number,localization_precision,sim_num))
np.savetxt(fname,np.transpose(hist_x_sphere),fmt='%.2f',delimiter=' ',newline='\n')

fname = file_path.replace('.npy','-D%.3f-bn%.0f-lo%.0f-sn%.0f-spherehist-y.txt' %(Dtrue,bin_number,localization_precision,sim_num))
np.savetxt(fname,np.transpose(hist_y_sphere),fmt='%.2f',delimiter=' ',newline='\n')

