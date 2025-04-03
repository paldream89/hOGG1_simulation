# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

# ================= 参数设置 =================
square_size = 1500  # 单位 nm
grid_bin = 15
slow_region_fraction = 0.44
D1_um = 0.2
D2_um = 2.5
time_interval = 0.006
steps = 100
N_track = 20  # 模拟轨迹数量

D1 = D1_um * 1e6  # nm²/s
D2 = D2_um * 1e6
t = time_interval / steps
square_size_half = square_size / 2

# 每个网格大小
cell_size = square_size / grid_bin

# 随机生成 slow 区索引
all_indices = [(i, j) for i in range(grid_bin) for j in range(grid_bin)]
num_slow = int(slow_region_fraction * len(all_indices))
slow_indices = set(random.sample(all_indices, num_slow))

# 判断函数
def is_in_slow_region(point):
    x, y = point
    i = int((x + square_size_half) // cell_size)
    j = int((y + square_size_half) // cell_size)
    return (i, j) in slow_indices if 0 <= i < grid_bin and 0 <= j < grid_bin else False

# ================= 可视化开始 =================
fig, ax = plt.subplots()

# 绘制边框
boundary = Rectangle((-square_size_half, -square_size_half), 
                     square_size, square_size, linewidth=1.5,
                     linestyle='--', edgecolor='#2F4F4F', 
                     facecolor='none', zorder=1)
ax.add_patch(boundary)

# 绘制网格背景
for i in range(grid_bin):
    for j in range(grid_bin):
        x0 = -square_size_half + i * cell_size
        y0 = -square_size_half + j * cell_size
        color = 'blue' if (i, j) in slow_indices else 'red'
        rect = Rectangle((x0, y0), cell_size, cell_size, facecolor=color,
                         edgecolor='none', alpha=0.8, zorder=0)
        ax.add_patch(rect)

# ================= 绘制 N_track 条轨迹 =================
for _ in range(N_track):
    trajectory_start = np.random.uniform(-square_size_half, square_size_half, 2)
    trajectory_current = trajectory_start.copy()
    trajectory_array = np.zeros((steps + 1, 2))
    trajectory_array[0] = trajectory_start

    for i in range(1, steps + 1):
        D = D1 if is_in_slow_region(trajectory_current) else D2
        sigma = np.sqrt(2 * D * t)
        trajectory_array[i] = trajectory_current + np.random.normal(0, sigma, 2)
        trajectory_current = trajectory_array[i]

    x, y = trajectory_array.T
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, steps), linewidth=3)
    lc.set_array(np.linspace(0, steps, len(segments)))
    ax.add_collection(lc)

    ax.scatter(x[0], y[0], s=240, color='lime', edgecolor='black', zorder=3)
    ax.scatter(x[-1], y[-1], s=240, color='red', edgecolor='black', zorder=3)

# 设置比例、坐标轴、图框
ax.set_aspect('equal')
ax.set_xlim(-square_size_half, square_size_half)
ax.set_ylim(-square_size_half, square_size_half)
ax.set_xticks([]), ax.set_yticks([])
ax.set_xticklabels([]), ax.set_yticklabels([])
for spine in ax.spines.values():
    spine.set_visible(False)

# 图像尺寸匹配区域大小
fig.set_size_inches(square_size / 150, square_size / 150)
plt.tight_layout(pad=0)
plt.show()
