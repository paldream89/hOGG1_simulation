import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from numba import njit, prange
import random
from math import atan2
from WriteSTORMBin_XcYcZZcFrame import write_storm_bin_XcYcZZcFrame
from WriteSTORMBin import Write_STORMbin
from ReadSTORMBin import read_storm_bin
import wx
from matplotlib.colors import ListedColormap

# ========== 模拟参数 ==========
square_size = 1500  # nm
grid_bin = 50
slow_region_fraction = 0.49
D1_um = 0.2
D2_um = 2.2
time_interval = 0.006
steps = 100
N_track = 20
N_binfile = 30000
pixel_size = 114
disp_thre = 5
loc_pre = 25 # unit is nm
binfile_xstart = 165.5
binfile_ystart = 41.5

D1 = D1_um * 1e6  # nm²/s
D2 = D2_um * 1e6
t = time_interval / steps
square_half = square_size / 2
cell_size = square_size / grid_bin

app = wx.App()
frame = wx.Frame(None, -1, 'win.py')
frame.SetSize(0,0,200,50)

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Select Multiple STORM Bin files", "", "", 
      "Bin files (*.bin)|*.bin", 
       wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)

openFileDialog.ShowModal()
file_path_list = openFileDialog.GetPaths()
openFileDialog.Destroy()

total_framebin, total_mol_numbin, original_mol_list = read_storm_bin(file_path_list[0])
total_framebin = total_framebin.item()
total_mol_numbin = total_mol_numbin.item()
# original_mol_list['Xc']  original_mol_list['Yc'] original_mol_list['Zc']

all_indices = [(i, j) for i in range(grid_bin) for j in range(grid_bin)]
num_slow = int(slow_region_fraction * len(all_indices))

# 输入数据
x = original_mol_list['Xc']
y = original_mol_list['Yc']
z = original_mol_list['Zc']

# 网格边长
grid_width = 13.15 / grid_bin

# 计算网格索引
ix = np.floor((x - binfile_xstart) / grid_width).astype(int)
iy = np.floor((y - binfile_ystart) / grid_width).astype(int)

# 过滤越界索引
valid_mask = (ix >= 0) & (ix < grid_bin) & (iy >= 0) & (iy < grid_bin)
ix = ix[valid_mask]
iy = iy[valid_mask]
z = z[valid_mask]

# 计算每个格子的Zc平均值
zc_sum = np.zeros((grid_bin, grid_bin), dtype=np.float64)
zc_count = np.zeros((grid_bin, grid_bin), dtype=np.int32)

for i in range(len(z)):
    zc_sum[ix[i], iy[i]] += z[i]
    zc_count[ix[i], iy[i]] += 1

with np.errstate(divide='ignore', invalid='ignore'):
    zc_avg = np.where(zc_count > 0, zc_sum / zc_count, np.nan)

#===========计算格子与相邻格子之间的zc差异值==========
# zc_diff = np.zeros_like(zc_avg)

# for i in range(grid_bin):
#     for j in range(grid_bin):
#         current = zc_avg[i, j]
#         neighbor_values = []

#         if i > 0 and not np.isnan(zc_avg[i - 1, j]):
#             neighbor_values.append(zc_avg[i - 1, j])
#         if i < grid_bin - 1 and not np.isnan(zc_avg[i + 1, j]):
#             neighbor_values.append(zc_avg[i + 1, j])
#         if j > 0 and not np.isnan(zc_avg[i, j - 1]):
#             neighbor_values.append(zc_avg[i, j - 1])
#         if j < grid_bin - 1 and not np.isnan(zc_avg[i, j + 1]):
#             neighbor_values.append(zc_avg[i, j + 1])

#         if neighbor_values:
#             neighbor_avg = np.mean(neighbor_values)
#             zc_diff[i, j] = abs(current - neighbor_avg)
#         else:
#             zc_diff[i, j] = 0
            
# # 可视化使用的 normalized prob_map
# zc_diff_safe = np.nan_to_num(zc_diff, nan=0.0)
# prob_map = (zc_diff_safe - np.min(zc_diff_safe)) / (np.max(zc_diff_safe) - np.min(zc_diff_safe) + 1e-8)
# prob_map = prob_map + 1e-6  # 防止为0
# selected = np.zeros_like(prob_map, dtype=bool)

#======动态随机====
# 归一化为 [0,1]，然后反转（值越小，概率越大）
zc_avg_safe = np.nan_to_num(zc_avg, nan=np.nanmax(zc_avg))  # 替换nan值
norm_zc = (zc_avg_safe - np.min(zc_avg_safe)) / (np.max(zc_avg_safe) - np.min(zc_avg_safe))
prob_map = 1.0 - norm_zc  # 越小值越容易被选中
prob_map = prob_map + 1e-6  # 防止为0
selected = np.zeros_like(prob_map, dtype=bool)

# 创建 slow_indices 结果容器
slow_indices = set()

def suppress_neighbors(prob, i, j, factor1=0.95,factor2=0.98):
    """抑制邻域概率：正交邻居用 factor1，斜对角邻居用 factor2"""
    neighbors_orthogonal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors_diagonal  = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 抑制正交邻居
    for dx, dy in neighbors_orthogonal:
        ni, nj = i + dx, j + dy
        if 0 <= ni < grid_bin and 0 <= nj < grid_bin and not selected[ni, nj]:
            prob[ni, nj] *= factor1

    # 抑制对角邻居
    for dx, dy in neighbors_diagonal:
        ni, nj = i + dx, j + dy
        if 0 <= ni < grid_bin and 0 <= nj < grid_bin and not selected[ni, nj]:
            prob[ni, nj] *= factor2

# 动态选择 num_slow 个格子
for _ in range(num_slow):
    flat_idx = np.argmax(prob_map * (~selected))  # 找最大概率 * 未选中掩码
    i, j = np.unravel_index(flat_idx, prob_map.shape)

    selected[i, j] = True
    slow_indices.add((i, j))

    # 抑制邻居
    suppress_neighbors(prob_map, i, j, factor1=0.95,factor2=0.98)

# 可视化：生成 slow_flags
slow_flags = np.zeros((grid_bin, grid_bin), dtype=bool)
for i, j in slow_indices:
    slow_flags[i, j] = True


#=========完全从小到大=========#
# 获取平均值最小的 num_slow 个格子
# flat_indices = np.argsort(np.nan_to_num(zc_avg, nan=np.inf), axis=None)
# slow_coords = np.unravel_index(flat_indices[:num_slow], (grid_bin, grid_bin))
# slow_indices = set(zip(slow_coords[0], slow_coords[1]))  # ✅ 类型匹配

# # 同时更新 slow_flags
# slow_flags = np.zeros((grid_bin, grid_bin), dtype=np.bool_)
# for i, j in slow_indices:
#     slow_flags[i, j] = True
    
#============仍然按一定概率======
# # 将 NaN 替换为最大值，防止影响排序
# zc_avg_safe = np.nan_to_num(zc_avg, nan=np.nanmax(zc_avg))

# # 归一化，并转换为反向权重（值越小，概率越大）
# zc_min, zc_max = np.min(zc_avg_safe), np.max(zc_avg_safe)
# normalized = (zc_avg_safe - zc_min) / (zc_max - zc_min + 1e-8)
# inverse_weight = 1 - normalized  # 0 ~ 1，越小Zc值，weight越大

# # 扁平化处理
# flat_weights = inverse_weight.flatten()
# flat_indices = np.arange(grid_bin * grid_bin)

# # 概率归一化
# probabilities = flat_weights / (np.sum(flat_weights) + 1e-8)

# # 从中选择 num_slow 个格子（带权采样）
# selected_flat = np.random.choice(flat_indices, size=num_slow, replace=False, p=probabilities)

# # 转换为 (i, j) 坐标
# slow_indices = set([(idx // grid_bin, idx % grid_bin) for idx in selected_flat])

# # 构建 slow_flags
# slow_flags = np.zeros((grid_bin, grid_bin), dtype=bool)
# for i, j in slow_indices:
#     slow_flags[i, j] = True

# ========== 生成 slow mask ==========
# slow_indices = set(random.sample(all_indices, num_slow))

# 用 mask 数组存储慢区信息
# for i, j in slow_indices:
#     slow_flags[i, j] = True

# 可视化 Zc 平均值热图
# 可视化 Zc 平均值热图（无坐标、标题、严格尺寸）
vmin_val = 0.0  # 设置你想要的最小值
vmax_val = 1.65  # 设置你想要的最大值
fig, ax = plt.subplots(figsize=(square_size / 150, square_size / 150))
ax.imshow(zc_avg_safe.T, origin='upper', cmap='jet', interpolation='nearest',
          vmin=vmin_val, vmax=vmax_val)

# 去掉坐标轴刻度和标签
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
for spine in ax.spines.values():
    spine.set_visible(False)

# 去掉标题和坐标名
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout(pad=0)
plt.show()

# 可视化 slow_flags 分布图
# 自定义颜色映射：False -> blue, True -> red
cmap = ListedColormap(['red', 'blue'])
plt.figure(figsize=(6, 5)) 
plt.imshow(slow_flags.T, origin='upper', cmap=cmap, interpolation='nearest')
plt.title('Selected Slow Region Mask')
plt.xlabel('Grid X')
plt.ylabel('Grid Y')
plt.tight_layout()
plt.show()

# ========== Numba 加速的运动函数 ==========
@njit(parallel=True)
def simulate_N_binfile(starts, slow_flags, square_size, grid_bin, D1, D2, t, steps):
    num_points = starts.shape[0]
    square_half = square_size / 2
    cell_size = square_size / grid_bin

    start_pt_array = np.zeros((num_points, 2), dtype=np.float32)
    end_pt_array = np.zeros((num_points, 2), dtype=np.float32)
    disp_array = np.zeros(num_points, dtype=np.float32)
    angle_array = np.zeros(num_points, dtype=np.float32)

    for idx in prange(num_points):
        start = starts[idx]
        current = start.copy()
        for _ in range(steps):
            x_shifted = current[0] + square_half
            y_shifted = current[1] + square_half
            i = int(x_shifted // cell_size)
            j = int(y_shifted // cell_size)
            D = D1
            if 0 <= i < grid_bin and 0 <= j < grid_bin:
                if slow_flags[i, j]:
                    D = D1
                else:
                    D = D2
            else:
                D = D2
            sigma = np.sqrt(2 * D * t)
            dx = np.random.normal(0, sigma)
            dy = np.random.normal(0, sigma)
            current[0] += dx
            current[1] += dy

        loc_uncertainty = np.random.normal(0, loc_pre, size=(4))
        start_pt_array[idx] = start + loc_uncertainty[0:2]
        # start_pt_array[2 * idx + 1] = current
        end_pt_array[idx] = current + loc_uncertainty[2:4]
        # end_pt_array[2 * idx + 1] = start
        disp = np.sqrt((end_pt_array[idx][0] - start_pt_array[idx][0]) ** 2 + \
                       (end_pt_array[idx][1] - start_pt_array[idx][1]) ** 2)
        angle = atan2(end_pt_array[idx][1] - start_pt_array[idx][1], 
                      end_pt_array[idx][0] - start_pt_array[idx][0])
        disp_array[idx] = disp
        # disp_array[2 * idx + 1] = disp
        angle_array[idx] = angle
        # angle_array[2 * idx + 1] = angle

    return start_pt_array, end_pt_array, disp_array, angle_array

# ========== 生成 N_binfile 个点的轨迹 ==========
starts = np.random.uniform(-square_half, square_half, size=(N_binfile, 2))
start_pt_array, end_pt_array, disp_array, angle_array = simulate_N_binfile(
    starts, slow_flags, square_size, grid_bin, D1, D2, t, steps
)

import pandas as pd
df = pd.DataFrame({
    "start_x": start_pt_array[:, 0],
    "start_y": start_pt_array[:, 1],
    "end_x": end_pt_array[:, 0],
    "end_y": end_pt_array[:, 1],
    "disp": disp_array,
    "angle": angle_array
})
# import ace_tools as tools; tools.display_dataframe_to_user(name="轨迹统计结果", dataframe=df)

# ========== 可视化部分（绘制 N_track 条轨迹） ==========
fig, ax = plt.subplots()
boundary = Rectangle((-square_half, -square_half), square_size, square_size,
                     linewidth=1.5, linestyle='--', edgecolor='#2F4F4F',
                     facecolor='none', zorder=1)
ax.add_patch(boundary)

# 画慢区背景色
for i in range(grid_bin):
    for j in range(grid_bin):
        x0 = -square_half + i * cell_size
        y0 = -square_half + j * cell_size
        color = 'blue' if slow_flags[i, j] else 'red'
        rect = Rectangle((x0, y0), cell_size, cell_size, facecolor=color,
                         edgecolor='none', alpha=0.8, zorder=0)
        ax.add_patch(rect)
        
def is_in_slow_region(point):
    x, y = point
    i = int((x + square_half) // cell_size)
    j = int((y + square_half) // cell_size)
    return (i, j) in slow_indices if 0 <= i < grid_bin and 0 <= j < grid_bin else False

# 随机生成并绘制 N_track 条轨迹
for _ in range(N_track):
    trajectory_start = np.random.uniform(-square_half, square_half, 2)
    trajectory_current = trajectory_start.copy()
    trajectory_array = np.zeros((steps + 1, 2))
    trajectory_array[0] = trajectory_start

    for i in range(1, steps + 1):
        D = D1 if is_in_slow_region(trajectory_current := trajectory_array[i-1]) else D2
        sigma = np.sqrt(2 * D * t)
        trajectory_array[i] = trajectory_current + np.random.normal(0, sigma, 2)

    x, y = trajectory_array.T
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, steps), linewidth=3)
    lc.set_array(np.linspace(0, steps, len(segments)))
    ax.add_collection(lc)
    ax.scatter(x[0], y[0], s=240, color='lime', edgecolor='black', zorder=3)
    ax.scatter(x[-1], y[-1], s=240, color='red', edgecolor='black', zorder=3)

ax.set_aspect('equal')
ax.set_xlim(-square_half, square_half)
ax.set_ylim(-square_half, square_half)
ax.invert_yaxis()
ax.set_xticks([]), ax.set_yticks([])
ax.set_xticklabels([]), ax.set_yticklabels([])
for spine in ax.spines.values():
    spine.set_visible(False)

fig.set_size_inches(square_size / 150, square_size / 150)
plt.tight_layout(pad=0)
plt.show()

#==============存贮bin文件==================
def interleave_arrays(arr1, arr2):
    """交替穿插两个一维数组"""
    if arr1.size != arr2.size:
        raise ValueError("输入数组长度必须相同")
    result = np.empty((arr1.size + arr2.size,), dtype=arr1.dtype)
    result[0::2] = arr1
    result[1::2] = arr2
    return result

disp_array = disp_array/pixel_size
indice = disp_array < disp_thre
start_pt_array = (start_pt_array[indice]+square_size)/pixel_size
end_pt_array = (end_pt_array[indice]+square_size)/pixel_size
disp_array = disp_array[indice]
angle_array = angle_array[indice]

total_number = np.sum(indice)
# frame_array = np.random.randint(low=1, 
#                            high=frame_num + 1,  # +1确保包含frame_num
#                            size=total_number)
# frame_array.sort()
# frame_array = np.linspace(1, frame_num+1, frame_num).astype(np.int16)
# frame_array = frame_array[indice]
frame_array = np.linspace(1, total_number, total_number).astype(np.int16)


# data_type = np.dtype([('x_start', np.float32), ('y_start', np.float32),
#                   ('x_end', np.float32), ('y_end', np.float32),
#                   ('frame', np.int32),
#                   ('disp', np.float32), ('angle', np.float32)])  

save_file_path = \
    r'E:\BaiduSyncdisk\sample_data\jingfang-fig125mw-2ms-4-5.4347ms_1\Simulate\M3\CT-IC%dDen-%.2f-%.1fD-%.1fD-dis%.1f-grid%.0f.bin'\
        % (N_binfile,slow_region_fraction,D1_um,D2_um,disp_thre,grid_bin) 
write_storm_bin_XcYcZZcFrame(save_file_path,total_number,total_number,
                              start_pt_array[:,0],start_pt_array[:,1],
                              end_pt_array[:,0],end_pt_array[:,1],
                              frame_array,disp_array,angle_array)
    
height_array = np.ones(total_number*2, dtype=np.float32)*206.031
area_array = np.ones(total_number*2, dtype=np.float32)*968.889
width_array = np.ones(total_number*2, dtype=np.float32)*279.095
phi_array = np.zeros(total_number*2, dtype=np.float32)
ax_array = np.ones(total_number*2, dtype=np.float32)
bg_array = np.ones(total_number*2, dtype=np.float32)*394.159
I_array = np.ones(total_number*2, dtype=np.float32)*1476.02
category_array = np.zeros(total_number*2, dtype=np.int32)
valid_array = np.zeros(total_number*2, dtype=np.int32)
length_array = np.ones(total_number*2, dtype=np.int32)
link_array = -np.ones(total_number*2, dtype=np.int32)

save_file_path = save_file_path[0:-4]+'-Zcdisp.bin'

xc_array = interleave_arrays(start_pt_array[:,0],end_pt_array[:,0])
yc_array = interleave_arrays(start_pt_array[:,1],end_pt_array[:,1])
z_array_angle = interleave_arrays(angle_array,angle_array)
zc_array_disp = interleave_arrays(disp_array,disp_array)
frame_array = np.linspace(1, 2*total_number, 2*total_number).astype(np.int32)

Write_STORMbin(save_file_path,total_number*2, 2*total_number,
                xc_array,
                yc_array, 
                xc_array,
                yc_array, 
                height_array, area_array, width_array, phi_array, ax_array, bg_array, 
                I_array, category_array, 
                valid_array,
                frame_array, 
                length_array, link_array, 
                z_array_angle, 
                zc_array_disp)
