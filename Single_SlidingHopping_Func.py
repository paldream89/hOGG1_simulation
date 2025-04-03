# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 07:56:26 2025

@author: xlm69
"""

import numpy as np
from math import floor,sqrt,e,pi,ceil
import random

def Single_SlidingHopping_Func(trajectory_start,square_size,grid_bin,slow_region_fraction,D1_um,
                               D2_um,time_interval,steps):
    
    # ================= 参数设置 =================
    # square_size = 1000 # unit is nm
    # grid_bin = 10 # 划分为100x100=10000小格
    # slow_region_fraction = 0.5 # 0-1之间的参数
    # D1_um = 0.2 # unit is um2/s
    # D2_um = 2 # unit is um2/s
    # time_interval = 0.006 # unit is second
    # steps = 100 # 模拟步数
    
    D1 = D1_um*10**6  # sliding区域扩散系数
    D2 = D2_um*10**6 # hopping区域扩散系数
    
    t = time_interval/steps  # 单步时间
    
    square_size_half = square_size/2
    
    # =========判断是否落入慢速区函数===========
    # 1. 每个网格的大小（单位 nm）
    cell_size = square_size / grid_bin
    
    # 2. 网格索引：总共 grid_bin x grid_bin 个格子，用(i,j)表示索引
    all_indices = [(i, j) for i in range(grid_bin) for j in range(grid_bin)]
    
    # 3. 从中随机选择 slow_region_fraction 比例的格子作为慢区
    num_slow = int(slow_region_fraction * len(all_indices))
    slow_indices = set(random.sample(all_indices, num_slow))  # 用 set 提高查找效率
    
    # 4. 判断 trajectory_start 是否落在某个慢区格子里
    def is_in_slow_region(point):
        """
        判断一个点是否落入慢区域。
        point: [x, y] 坐标（单位 nm）
        """
        x = point[0]
        y = point[1]
        
        # 平移坐标系到从 (0,0) 开始
        x_shifted = x + square_size_half
        y_shifted = y + square_size_half
    
        # 获取其所属的格子索引（i,j）
        i = int(x_shifted // cell_size)
        j = int(y_shifted // cell_size)
    
        # 越界检查
        if 0 <= i < grid_bin and 0 <= j < grid_bin:
            return (i, j) in slow_indices
        else:
            return False
    
    trajectory_current = trajectory_start
    
    # ================= 初始化坐标 =================
    # 生成初始坐标（使用numpy提高效率）
    
    # ================= 动态扩散模拟 =================
    for i in range(1, steps+1):
        
        # 动态判断扩散系数（完全向量化判断）
        D = D1 if is_in_slow_region(trajectory_current) else D2
        
        # 计算位移标准差（基于当前D值）
        sigma = np.sqrt(2 * D * t)
        
        # 生成位移并更新坐标（向量化操作）
        trajectory_current = trajectory_current + np.random.normal(0, sigma, 2)
        
    return trajectory_current
    