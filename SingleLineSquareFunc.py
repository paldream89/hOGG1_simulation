# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 07:56:26 2025

@author: xlm69
"""
import numpy as np

def SingleLineSquareFunc(start_pt,pixel_size,slow_rangenm,
                         D1_um,D2_um,time_interval,steps,square_size):
    
    # ================= 参数设置 =================
    # pixel_size = 110 # unit is nm
    # slow_rangenm = 50 # unit is nm
    # # loc_precision_nm = 10 # unit is nm
    # D1_um = 0.4 # unit is um2/s
    # D2_um = 2 # unit is um2/s
    # time_interval = 0.005 # unit is second
    # steps = 100 # 模拟步数
    # square_size = 5 # size of simulation
    
    # loc_precision = loc_precision_nm/pixel_size
    D1 = D1_um*10**6/pixel_size/pixel_size  # 中心区域扩散系数
    D2 = D2_um*10**6/pixel_size/pixel_size  # 外围区域扩散系数
    t = time_interval/steps  # 单步时间
    slow_range = slow_rangenm/pixel_size/2
    # square_size_half = square_size/2
    # start_pt = np.random.uniform(-square_size_half, square_size_half, (2))
    
    # ================= 初始化坐标 =================
    # 生成初始坐标（使用numpy提高效率）
    
    trajectory = start_pt # 生成初始坐标并直接创建数组
    
    # ================= 动态扩散模拟 =================
    for i in range(1, steps+1):
        current_x = trajectory[0]
        # 动态判断扩散系数（完全向量化判断）
        D = D1 if (-slow_range <= current_x <= slow_range) else D2
        
        # 计算位移标准差（基于当前D值）
        sigma = np.sqrt(2 * D * t)
        
        # 生成位移并更新坐标（向量化操作）
        trajectory = trajectory + np.random.normal(0, sigma, 2)
        
    end_pt = trajectory
    # random_array = np.random.normal(loc=0, scale=loc_precision, size=4)
    # result = np.concatenate((start_pt,end_pt),axis=0)
    # result = result+random_array
    
    return end_pt

def DoubleLineSquareFunc(start_pt,pixel_size,slow_rangenm,slow_sepa,
                         D1_um,D2_um,time_interval,steps,square_size):
    
    # ================= 参数设置 =================
    # pixel_size = 110 # unit is nm
    # slow_rangenm = 50 # unit is nm
    # # loc_precision_nm = 10 # unit is nm
    # D1_um = 0.4 # unit is um2/s
    # D2_um = 2 # unit is um2/s
    # time_interval = 0.005 # unit is second
    # steps = 100 # 模拟步数
    # square_size = 5 # size of simulation
    
    # loc_precision = loc_precision_nm/pixel_size
    D1 = D1_um*10**6/pixel_size/pixel_size  # 中心区域扩散系数
    D2 = D2_um*10**6/pixel_size/pixel_size  # 外围区域扩散系数
    t = time_interval/steps  # 单步时间
    sep = slow_sepa/pixel_size/2
    wid = slow_rangenm/pixel_size/2
    # square_size_half = square_size/2
    # start_pt = np.random.uniform(-square_size_half, square_size_half, (2))
    
    # ================= 初始化坐标 =================
    # 生成初始坐标（使用numpy提高效率）
    
    trajectory = start_pt # 生成初始坐标并直接创建数组
    
    # ================= 动态扩散模拟 =================
    for i in range(1, steps+1):
        current_x = trajectory[0]
        # 动态判断扩散系数（完全向量化判断）
        D = D1 if ((sep - wid <= current_x <= sep + wid) or (-sep - wid <= current_x <= -sep + wid)) else D2
        
        # 计算位移标准差（基于当前D值）
        sigma = np.sqrt(2 * D * t)
        
        # 生成位移并更新坐标（向量化操作）
        trajectory = trajectory + np.random.normal(0, sigma, 2)
        
    end_pt = trajectory
    # random_array = np.random.normal(loc=0, scale=loc_precision, size=4)
    # result = np.concatenate((start_pt,end_pt),axis=0)
    # result = result+random_array
    
    return end_pt

def SingleCircleSquareFunc(start_pt,pixel_size,slow_rangenm,
                         D1_um,D2_um,time_interval,steps,square_size):
    
    # ================= 参数设置 =================
    # pixel_size = 110 # unit is nm
    # slow_rangenm = 50 # unit is nm
    # # loc_precision_nm = 10 # unit is nm
    # D1_um = 0.4 # unit is um2/s
    # D2_um = 2 # unit is um2/s
    # time_interval = 0.005 # unit is second
    # steps = 100 # 模拟步数
    # square_size = 5 # size of simulation
    
    # loc_precision = loc_precision_nm/pixel_size
    D1 = D1_um*10**6/pixel_size/pixel_size  # 中心区域扩散系数
    D2 = D2_um*10**6/pixel_size/pixel_size  # 外围区域扩散系数
    t = time_interval/steps  # 单步时间
    slow_range = slow_rangenm/pixel_size/2
    # square_size_half = square_size/2
    # start_pt = np.random.uniform(-square_size_half, square_size_half, (2))
    
    # ================= 初始化坐标 =================
    # 生成初始坐标（使用numpy提高效率）
    
    trajectory = start_pt # 生成初始坐标并直接创建数组
    
    # ================= 动态扩散模拟 =================
    for i in range(1, steps+1):
        current_x = trajectory[0]
        current_y = trajectory[1]
        distance = (current_x**2+current_y**2)**0.5
        # 动态判断扩散系数（完全向量化判断）
        D = D1 if (distance <= slow_range) else D2
        
        # 计算位移标准差（基于当前D值）
        sigma = np.sqrt(2 * D * t)
        
        # 生成位移并更新坐标（向量化操作）
        trajectory = trajectory + np.random.normal(0, sigma, 2)
        
    end_pt = trajectory
    # random_array = np.random.normal(loc=0, scale=loc_precision, size=4)
    # result = np.concatenate((start_pt,end_pt),axis=0)
    # result = result+random_array
    
    return end_pt

def DoubleLineSquareFunc(start_pt,pixel_size,slow_rangenm,slow_sepa,
                         D1_um,D2_um,time_interval,steps,square_size):
    
    # ================= 参数设置 =================
    # pixel_size = 110 # unit is nm
    # slow_rangenm = 50 # unit is nm
    # # loc_precision_nm = 10 # unit is nm
    # D1_um = 0.4 # unit is um2/s
    # D2_um = 2 # unit is um2/s
    # time_interval = 0.005 # unit is second
    # steps = 100 # 模拟步数
    # square_size = 5 # size of simulation
    
    # loc_precision = loc_precision_nm/pixel_size
    D1 = D1_um*10**6/pixel_size/pixel_size  # 中心区域扩散系数
    D2 = D2_um*10**6/pixel_size/pixel_size  # 外围区域扩散系数
    t = time_interval/steps  # 单步时间
    sep = slow_sepa/pixel_size/2
    wid = slow_rangenm/pixel_size/2
    # square_size_half = square_size/2
    # start_pt = np.random.uniform(-square_size_half, square_size_half, (2))
    
    # ================= 初始化坐标 =================
    # 生成初始坐标（使用numpy提高效率）
    
    trajectory = start_pt # 生成初始坐标并直接创建数组
    
    # ================= 动态扩散模拟 =================
    for i in range(1, steps+1):
        current_x = trajectory[0]
        # 动态判断扩散系数（完全向量化判断）
        D = D1 if ((sep - wid <= current_x <= sep + wid) or (-sep - wid <= current_x <= -sep + wid)) else D2
        
        # 计算位移标准差（基于当前D值）
        sigma = np.sqrt(2 * D * t)
        
        # 生成位移并更新坐标（向量化操作）
        trajectory = trajectory + np.random.normal(0, sigma, 2)
        
    end_pt = trajectory
    # random_array = np.random.normal(loc=0, scale=loc_precision, size=4)
    # result = np.concatenate((start_pt,end_pt),axis=0)
    # result = result+random_array
    
    return end_pt






# # ================= 科学可视化 =================
# plt.figure(figsize=(square_size+1, square_size+1))  # 关键修改：正方形画布
# ax = plt.gca()

# # 绘制5x5边界框（精确对齐坐标）
# boundary = Rectangle((-square_size_half, -square_size_half), 
#                      square_size, square_size, linewidth=1.5,
#                     linestyle='--', edgecolor='#2F4F4F', 
#                     facecolor='none', zorder=1)
# ax.add_patch(boundary)

# # 绘制判定边界（使用精确坐标对齐）
# ax.axvline(-slow_range, color='tab:gray', linestyle=':', alpha=0.8, lw=1.2)
# ax.axvline(slow_range, color='tab:gray', linestyle=':', alpha=0.8, lw=1.2)

# # 绘制运动轨迹（带颜色渐变效果）
# x, y = trajectory.T
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)

# lc = LineCollection(segments, cmap='viridis', 
#                    norm=plt.Normalize(0, steps), linewidth=1.5)
# lc.set_array(np.linspace(0, steps, len(segments)))
# ax.add_collection(lc)

# # 绘制起点终点标记
# ax.scatter(x[0], y[0], s=80, color='lime', edgecolor='black', 
#           zorder=4, label='Start')
# ax.scatter(x[-1], y[-1], s=80, color='red', edgecolor='black',
#           zorder=4, label='End')

# # 设置科学绘图比例
# ax.set_aspect('equal')  # 关键修改：强制等比例坐标
# ax.set_xlim(-ceil(square_size_half), ceil(square_size_half))
# ax.set_ylim(-ceil(square_size_half), ceil(square_size_half))
# ax.xaxis.set_tick_params(which='both', direction='in')
# ax.yaxis.set_tick_params(which='both', direction='in')

# # 添加比例尺（可选）
# scale_bar = Rectangle((2, -2.8), 0.5, 0.15, color='black')
# ax.add_patch(scale_bar)
# ax.text(2, -2.7, '0.5 unit', ha='left', va='bottom')

# # 添加图例和标题
# ax.legend(loc='upper right', framealpha=0.9)
# plt.title(f"2D Brownian Motion\nD1={D1}, D2={D2} ", pad=15)
# plt.tight_layout()
# plt.show()