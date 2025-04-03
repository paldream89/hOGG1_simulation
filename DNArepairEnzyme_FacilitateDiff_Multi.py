# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:09:39 2024

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:08:26 2024

@author: Admin
"""

# simulation for DNA repair enzyme facilitated diffusion
import matplotlib.pyplot as plt
import numpy as np
import time
from math import floor,sqrt,e,pi
from scipy import optimize
import matplotlib as mpl
from matplotlib.pyplot import hist
import random

# base pair number setting
Total_bp = 10**6 # total DNA base pair number
OxoG_per = 0.0001 # percentage of 8-oxo-G
 # 1 in 10^5 to 10^6 G to oxoG under normal 
Bp_distance = 0.34 # unit is nm, DNA base pair distance
Chromosome_DNA_density = 3.2*10**9/5000 # unit is bp per um3
# 3.2*10**9/500 is 6.4 × 10^6 base pairs per μm³
# suggested a density of ~2 × 10^8 base pairs per μm³

# Diffusion setting
Total_time = 0.01 # unit is second. So 0.001 means 1 ms
Slide_per_num = 1
Slide_per_array = np.linspace(0.98,0.98,num=Slide_per_num)
# Slide_per = 0.75 # sliding percentage
# Hop_per = 1-Slide_per # hopping percentage
D_slide = 0.58 # unit is um2/s
D_hop = 2 # unit is um2/s

# How many runs, then get array and average
Runs = 100

# Calculation and preparation
Total_bp_array = np.linspace(0,Total_bp-1,num=Total_bp,dtype=np.int32)
Num_oxoG = int(Total_bp*OxoG_per)
# Indicator_array = np.zeros(Num_oxoG)

# One loop, first slide, check to see if reach oxoG, put its indice to 1
# Then hop for a long distance
start_time = time.time()
results_array = np.zeros((Runs,Slide_per_num),dtype=np.float32)

j=0
while j<Slide_per_num:

    i=0
    Slide_per = Slide_per_array[j]
    Hop_per = 1-Slide_per
    A_loop_num_array = np.zeros(Runs,dtype=np.float32)
    Gauss_sigma = sqrt(2*D_slide*Total_time*Slide_per*1000*1000) # unit is nm
    Gauss_sigma_hop = sqrt(6*D_hop*Total_time*Hop_per) # unit is um
    
    while i<Runs:
        
        A_loop_num = 1
        # OxoG_indice_array = np.int32(np.floor(np.random.rand(Num_oxoG)*Total_bp))
        OxoG_indice_array = np.random.choice(Total_bp_array,size=Num_oxoG)
        Start_bp_indice = floor(random.random()*Total_bp)
    
        while np.all(OxoG_indice_array==-1)==False:
            
            # Slide
            Slide_bp_num = int(np.random.normal(0,Gauss_sigma,1)[0]/Bp_distance)
            End_bp_indice = Start_bp_indice+Slide_bp_num
            
            if End_bp_indice>=0 and End_bp_indice<Total_bp:
                
                # do not reach the boundary
                if  Slide_bp_num>=0: # Slide to larger direction
                    Checked_bp = (np.linspace(Start_bp_indice,End_bp_indice,num=Slide_bp_num+1,dtype=np.int32))
                    mask = np.isin(OxoG_indice_array,Checked_bp)
                    # 前一个数列是否在后面一个，输出前一个数列对应的索引
                    OxoG_indice_array[mask]=-1
                    
                else: # Slide to smaller direction
                    Checked_bp = (np.linspace(End_bp_indice,Start_bp_indice,num=-Slide_bp_num+1,dtype=np.int32))
                    mask = np.isin(OxoG_indice_array,Checked_bp)
                    # 前一个数列是否在后面一个，输出前一个数列对应的索引
                    OxoG_indice_array[mask]=-1
                    
            elif End_bp_indice<0: # Slide to smaller direction且超过了0
                
                End_bp_indice = End_bp_indice+Total_bp 
                Checked_bp = (np.linspace(0,Start_bp_indice,num=Start_bp_indice+1,dtype=np.int32))
                mask = np.isin(OxoG_indice_array,Checked_bp)
                # 前一个数列是否在后面一个，输出前一个数列对应的索引
                OxoG_indice_array[mask]=-1
                Checked_bp = (np.linspace(End_bp_indice,Total_bp-1,num=Total_bp-End_bp_indice,dtype=np.int32))
                mask = np.isin(OxoG_indice_array,Checked_bp)
                # 前一个数列是否在后面一个，输出前一个数列对应的索引
                OxoG_indice_array[mask]=-1
                
            elif End_bp_indice>=Total_bp: # Slide to lager direction且超过了Total_bp
            
                End_bp_indice = End_bp_indice-Total_bp 
                Checked_bp = (np.linspace(0,End_bp_indice,num=End_bp_indice+1,dtype=np.int32))
                mask = np.isin(OxoG_indice_array,Checked_bp)
                # 前一个数列是否在后面一个，输出前一个数列对应的索引
                OxoG_indice_array[mask]=-1
                Checked_bp = (np.linspace(Start_bp_indice,Total_bp-1,num=Total_bp-Start_bp_indice,dtype=np.int32))
                mask = np.isin(OxoG_indice_array,Checked_bp)
                # 前一个数列是否在后面一个，输出前一个数列对应的索引
                OxoG_indice_array[mask]=-1
            
            A_loop_num+=1
            Start_bp_indice = End_bp_indice
            
            # Hop
            Hop_bp_num = int((4/3*pi*(np.random.normal(0,Gauss_sigma_hop,1)[0]/2)**3)*Chromosome_DNA_density)
            End_bp_indice = Start_bp_indice+Hop_bp_num
                    
            if End_bp_indice<0: # Hop to smaller direction且超过了0
                
                while End_bp_indice<0: 
                    End_bp_indice = End_bp_indice+Total_bp
                
            elif End_bp_indice>=Total_bp: # Hop to lager direction且超过了Total_bp
            
                while End_bp_indice>=Total_bp: 
                    End_bp_indice = End_bp_indice-Total_bp 
                    
            Start_bp_indice = End_bp_indice
        
        A_loop_num_array[i]=A_loop_num*Total_time
        i+=1
        print(str(i))
    
    results_array[:,j] = A_loop_num_array
    j+=1
    

print("--- %s seconds ---" % (time.time() - start_time))

        






