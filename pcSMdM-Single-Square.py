# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 07:56:26 2025

@author: xlm69
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# ================= 参数设置 =================
pixel_size = 114  # unit is nm
slow_rangenm = 0  # unit is nm

D1_um = 2  # unit is um2/s
D2_um = 2  # unit is um2/s
time_interval = 0.005  # unit is second
num_molecule = 100  # 分子数量

D1 = D1_um * 10**6 / pixel_size / pixel_size  # 中心区域扩散系数
D2 = D2_um * 10**6 / pixel_size / pixel_size  # 外围区域扩散系数
steps = 1  # 模拟步数
t = time_interval / steps  # 单步时间

slow_range = slow_rangenm / pixel_size / 2
square_size = 7.0175438596491228070175438596491  # size of simulation
square_size_half = square_size / 2

# ================= 初始化坐标 =================
# 生成 num_molecule 个初始坐标
start_points = np.random.uniform(-square_size_half, square_size_half, (num_molecule, 2))

# ================= 动态扩散模拟 =================
def simulate_brownian_motion(start_point):
    trajectory = np.array([start_point])  # 初始化轨迹
    for _ in range(steps):
        current_x = trajectory[-1, 0]
        # 动态判断扩散系数
        D = D1 if (-slow_range <= current_x <= slow_range) else D2

        # 计算位移标准差
        sigma = np.sqrt(2 * D * t)

        # 生成位移并更新坐标
        displacement = np.random.normal(0, sigma, 2)
        new_point = trajectory[-1] + displacement
        trajectory = np.append(trajectory, [new_point], axis=0)
    return trajectory

# 模拟所有分子的运动轨迹
all_trajectories = [simulate_brownian_motion(start_point) for start_point in start_points]

# ================= 科学可视化 =================
# 设置画布大小与坐标轴长度完全吻合
fig, ax = plt.subplots(figsize=(square_size, square_size))  # 画布大小与坐标轴长度一致

# 绘制所有分子的运动轨迹
for trajectory in all_trajectories:
    x, y = trajectory.T
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='viridis',
                       norm=plt.Normalize(0, steps), linewidth=1.5)
    lc.set_array(np.linspace(0, steps, len(segments)))
    ax.add_collection(lc)

    # 绘制起点和终点
    ax.scatter(x[0], y[0], s=80, color='lime', edgecolor='black', zorder=4)
    ax.scatter(x[-1], y[-1], s=80, color='red', edgecolor='black', zorder=4)

# 设置科学绘图比例
ax.set_aspect('equal')  # 强制等比例坐标

# 设置坐标范围
ax.set_xlim(-square_size_half, square_size_half)
ax.set_ylim(-square_size_half, square_size_half)

# 隐藏坐标轴
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 保存图像
plt.savefig("brownian_motion_multiple.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()