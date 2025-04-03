# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:49:20 2023

@author: Admin
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from math import floor,sqrt,e,pi,cos,sin,acos,atan2
from scipy import optimize
import matplotlib as mpl
import matplotlib.path as mpltPath
from matplotlib.pyplot import hist
import random
from Reflections import reflect_circle
# from Reflections import reflect_circle

# displacement in a sphere generator

def sphere_disp_generator_func(sphere_diameter,D,bin_number,localization_precision,sim_num):
    
    # bin_number = 20
    # D = 4 # unit is um2/s
    # sphere_diameter = 300 # unit is nm
    # localization_precision = 30 # unit is nm
    # sim_num = 100 # must be even number!!!
    
    divide_num = 1000 # the smaller, the better, maybe 1000?
    time_interval = 0.001 # unit is s, so 0.001 is 1ms
    Gauss_sigma = sqrt(2*D*time_interval/divide_num*1000*1000)
    sphere_radius = sphere_diameter/2 # unit is nm
    # switch_sphere_noboundary = 1 # 0 no boundary, 1 circle
    round_digit = 10000 # 10000:保留小数点后四位；100：两位
    gold = 0.6180339887
    rand_num = 1  # do NOT change
    bin_range = sphere_diameter+2*localization_precision # from 0
    if bin_range > 300:
        bin_range = 300
    
    def twoD_diff_dis_func(x,a,b):
        
        return 2*b*x/a*np.exp(-x**2/a)
    
    plt.close(fig='all')
    
    x_end_array = np.zeros(sim_num)
    y_end_array = np.zeros(sim_num)
    z_end_array = np.zeros(sim_num)
    x_disp_array = np.zeros(sim_num)
    y_disp_array = np.zeros(sim_num)
    x_start_array = np.zeros(sim_num)
    y_start_array = np.zeros(sim_num)
    z_start_array = np.zeros(sim_num)
    
    x_individual_array = np.zeros(divide_num)
    y_individual_array = np.zeros(divide_num)
    z_individual_array = np.zeros(divide_num)
    x_individual_disp_array = np.zeros(divide_num)
    y_individual_disp_array = np.zeros(divide_num)
    
    indice = int(-sim_num/2)+1
    counter = 0
    start_time = time.time()
    
    while counter < sim_num:
        
        z_start = (2*indice-1)/(sim_num-1)
        x_start = sqrt(1-z_start**2)*cos(2*pi*(indice)*gold)
        y_start = sqrt(1-z_start**2)*sin(2*pi*(indice)*gold)
        z_start = z_start*sphere_radius
        x_start = x_start*sphere_radius
        y_start = y_start*sphere_radius
        z_start_array[counter] = z_start
        x_start_array[counter] = x_start
        y_start_array[counter] = y_start
        
        theta0 = acos(z_start/sphere_radius) # should be 0 to pi
        phi0 = atan2(y_start,x_start) # should be -pi to pi
        counter_divide = 0
        
        while counter_divide<divide_num:
            
            x_disp = np.random.normal(0,Gauss_sigma,rand_num)
            y_disp = np.random.normal(0,Gauss_sigma,rand_num)
            theta_disp = x_disp[0]/sphere_radius
            theta1 = theta0+theta_disp
            if theta0==0:
                theta_temp = 1/round_digit
            else:
                theta_temp = theta0
            phi_disp = y_disp[0]/sphere_radius/sin(theta_temp)
            
            if theta1>=pi or theta1<0:
                
                if floor(theta1/pi)%2==1:
                    theta1 = (floor(theta1/pi)+1)*pi-theta1
                else:
                    if floor(theta1/pi)%2==0:
                        theta1 = theta1-floor(theta1/pi)*pi
                    
            phi1 = phi0+phi_disp
            if phi1>pi or phi1<=-pi:
                
                phi1 = phi1-floor((phi1+pi)/2/pi)*2*pi
                
            # error checking
            if theta1-theta0>pi/2:
                
                print('theta0: %.3f;  phi0: %.3f; theta1: %.3f;  phi1: %.3f; ' %(theta0,phi0,theta1,phi1))
                print('theta_disp: %.3f;  phi_disp: %.3f;' %(theta_disp,phi_disp))
                
            theta1 = np.fix(theta1*round_digit)/round_digit
            phi1 = np.fix(phi1*round_digit)/round_digit
            theta0 = theta1
            phi0 = phi1
            
            x_individual_array[counter_divide]=sphere_radius*sin(theta1)*cos(phi1)
            y_individual_array[counter_divide]=sphere_radius*sin(theta1)*sin(phi1)
            z_individual_array[counter_divide]=sphere_radius*cos(theta1)
            
            counter_divide+=1
        
        theta_end = theta1
        phi_end = phi1
        x_end_array[counter]=sphere_radius*sin(theta_end)*cos(phi_end)
        y_end_array[counter]=sphere_radius*sin(theta_end)*sin(phi_end)
        
        counter+=1
        indice+=1
        
        
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(projection='3d')
    # plot_num = 10000
    # plot_indice = int(-plot_num/2)+1
    # plot_counter = 0
    # x_plot_array = np.zeros(plot_num)
    # y_plot_array = np.zeros(plot_num)
    # z_plot_array = np.zeros(plot_num)
    # while plot_counter < plot_num:
        
    #     z_plot = (2*plot_indice-1)/(plot_num-1)
    #     x_plot = sqrt(1-z_plot**2)*cos(2*pi*(plot_indice)*gold)
    #     y_plot = sqrt(1-z_plot**2)*sin(2*pi*(plot_indice)*gold)
    #     z_plot = z_plot*sphere_radius
    #     x_plot = x_plot*sphere_radius
    #     y_plot = y_plot*sphere_radius
    #     z_plot_array[plot_counter] = z_plot
    #     x_plot_array[plot_counter] = x_plot
    #     y_plot_array[plot_counter] = y_plot
    #     plot_counter+=1
    #     plot_indice+=1
        
    # ax.scatter(x_plot_array, y_plot_array, z_plot_array,s=1,c='b')
    # ax.plot(x_individual_array, y_individual_array, z_individual_array,c='r')
    
    # localization precision added
    x_disp_array = x_end_array-x_start_array
    y_disp_array = y_end_array-y_start_array
    
    x_disp_loc_array = np.random.normal(0,localization_precision,sim_num)
    y_disp_loc_array = np.random.normal(0,localization_precision,sim_num)
    x_disp_array = x_disp_array+x_disp_loc_array
    y_disp_array = y_disp_array+y_disp_loc_array
    
    plt.figure(figsize=(8,8))
    xy_disp_array = np.sqrt(x_disp_array**2+y_disp_array**2)
    disp_hist = hist(xy_disp_array,bins=bin_number,range=(0,bin_range),rwidth=0.8)
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
        
        smooth_x = np.linspace(0,bin_range,num=int(100))
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
    
    return x_results,y_results,selected_drate




    