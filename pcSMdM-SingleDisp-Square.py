import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.cm as cm
from scipy.stats import norm
from SingleLineSquareFunc import SingleLineSquareFunc  # 需确认函数实现
from WriteSTORMBin_XcYcZZcFrame import write_storm_bin_XcYcZZcFrame
from WriteSTORMBin import Write_STORMbin


# ================= 参数设置 =================
pixel_size = 114  # unit is nm
slow_rangenm = 0  # unit is nm

D1_um = 2  # unit is um2/s
D2_um = 2  # unit is um2/s
time_interval = 0.005  # unit is second
num_molecule = 10  # 分子数量

D1 = D1_um * 10**6 / pixel_size / pixel_size  # 中心区域扩散系数
D2 = D2_um * 10**6 / pixel_size / pixel_size  # 外围区域扩散系数
steps = 1  # 模拟步数
t = time_interval / steps  # 单步时间

slow_range = slow_rangenm / pixel_size / 2
square_size = 7.0175438596491228070175438596491  # size of simulation
square_size_half = square_size / 2

# 设定颜色映射的范围
lower_limit = 0  # 设置下限
upper_limit = 3.5   # 设置上限

# ================= 初始化坐标 =================
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
    return trajectory, np.linalg.norm(trajectory[-1] - trajectory[0])  # 返回轨迹和位移

# 模拟所有分子的运动轨迹
all_data = [simulate_brownian_motion(start_point) for start_point in start_points]
all_trajectories = [data[0] for data in all_data]
displacements = np.array([data[1] for data in all_data])  # 计算位移

# 归一化位移用于颜色映射
displacements = np.clip(displacements, lower_limit, upper_limit)
colors = (displacements - lower_limit) / (upper_limit - lower_limit)

# 预先分配内存，提高计算效率
x_array = np.empty(num_molecule * 2)
y_array = np.empty(num_molecule * 2)
disp_array = np.empty(num_molecule * 2)

# 填充数据
for i, trajectory in enumerate(all_trajectories):
    start_x, start_y = trajectory[0]  # 起点坐标
    end_x, end_y = trajectory[-1]  # 终点坐标
    displacement = displacements[i]  # 对应的位移值

    # 直接索引存储数据，避免 append 操作
    x_array[2 * i] = start_x
    x_array[2 * i + 1] = end_x
    y_array[2 * i] = start_y
    y_array[2 * i + 1] = end_y
    disp_array[2 * i] = displacement
    disp_array[2 * i + 1] = displacement

# 输出数组检查

    save_file_path = \
        r'E:\BaiduSyncdisk\Python_programs\pcSMdM-simulation\ForFigureConvol.bin'\
        
    height_array = np.ones(num_molecule*2, dtype=np.float32)*206.031
    area_array = np.ones(num_molecule*2, dtype=np.float32)*968.889
    width_array = np.ones(num_molecule*2, dtype=np.float32)*279.095
    phi_array = np.zeros(num_molecule*2, dtype=np.float32)
    ax_array = np.ones(num_molecule*2, dtype=np.float32)
    bg_array = np.ones(num_molecule*2, dtype=np.float32)*394.159
    I_array = np.ones(num_molecule*2, dtype=np.float32)*1476.02
    category_array = np.zeros(num_molecule*2, dtype=np.int32)
    valid_array = np.zeros(num_molecule*2, dtype=np.int32)
    length_array = np.ones(num_molecule*2, dtype=np.int32)
    link_array = -np.ones(num_molecule*2, dtype=np.int32)
    
    save_file_path = save_file_path[0:-4]+'-Zcdisp.bin'
    
    frame_array = np.linspace(1, num_molecule*2, num_molecule*2).astype(np.int16)

    z_array_angle = np.zeros(num_molecule*2, dtype=np.float32)

    Write_STORMbin(save_file_path,num_molecule*2, 2*num_molecule,
                    x_array+square_size,
                    square_size-y_array, 
                    x_array+square_size,
                    square_size-y_array, 
                    height_array, area_array, width_array, phi_array, ax_array, bg_array, 
                    I_array, category_array, 
                    valid_array,
                    frame_array, 
                    length_array, link_array, 
                    z_array_angle, 
                    upper_limit-disp_array)


# ================= 科学可视化 =================
# ================= 保存第一张图（原始布朗运动图，黑色连线+黑色边框实心圆） =================
fig, ax1 = plt.subplots(figsize=(square_size, square_size))
for trajectory in all_trajectories:
    x, y = trajectory.T
    ax1.plot(x, y, color='black', alpha=0.5, linewidth=1)  # 纯黑色连线
    ax1.scatter(x[0], y[0], s=80, color='lime', edgecolor='black', linewidth=1, zorder=4)  # 起点实心圆
    ax1.scatter(x[-1], y[-1], s=80, color='red', edgecolor='black', linewidth=1, zorder=4)  # 终点实心圆

ax1.set_aspect('equal')
ax1.set_xlim(-square_size_half, square_size_half)
ax1.set_ylim(-square_size_half, square_size_half)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

plt.savefig("brownian_motion_original.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

# ================= 保存第二张图（起点+终点，颜色变化但无连线） =================
fig, ax2 = plt.subplots(figsize=(square_size, square_size))
for i, trajectory in enumerate(all_trajectories):
    x, y = trajectory.T
    # 起点颜色按照位移映射
    ax2.scatter(x[0], y[0], s=80, color=plt.cm.jet(colors[i]), edgecolor='black', linewidth=1, zorder=4)  # 起点实心圆
    ax2.scatter(x[-1], y[-1], s=80, color=plt.cm.jet(colors[i]), edgecolor='black', linewidth=1, zorder=4)  # 终点实心圆

ax2.set_aspect('equal')
ax2.set_xlim(-square_size_half, square_size_half)
ax2.set_ylim(-square_size_half, square_size_half)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.savefig("brownian_motion_colored_with_start.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

# ================= 保存第三张图（起点+终点+网格） =================
fig, ax3 = plt.subplots(figsize=(square_size, square_size))

# 绘制起点和终点（颜色映射相同）
for i, trajectory in enumerate(all_trajectories):
    x, y = trajectory.T
    ax3.scatter(x[0], y[0], s=80, color=plt.cm.jet(colors[i]), edgecolor='black', linewidth=1, zorder=4)  # 起点实心圆
    ax3.scatter(x[-1], y[-1], s=80, color=plt.cm.jet(colors[i]), edgecolor='black', linewidth=1, zorder=4)  # 终点实心圆

# 添加网格（8x8均匀划分）
num_grid = 3
x_ticks = np.linspace(-square_size_half, square_size_half, num_grid + 1)
y_ticks = np.linspace(-square_size_half, square_size_half, num_grid + 1)
ax3.set_xticks(x_ticks)
ax3.set_yticks(y_ticks)
ax3.grid(which='both', linestyle='--', linewidth=3, color='black', alpha=1)

# 其他图像设置
ax3.set_aspect('equal')
ax3.set_xlim(-square_size_half, square_size_half)
ax3.set_ylim(-square_size_half, square_size_half)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

# 保存图像
plt.savefig("brownian_motion_colored_with_grid.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()


# # ================= 重新生成第四张图（正态分布渐变透明圆形，独立参数+修正透明度范围） =================
# fig, ax4 = plt.subplots(figsize=(square_size, square_size))

# # 设置圆的最大半径（单位与坐标轴一致，可调节）
# max_radius = 0.5  # 设定最大圆半径

# # 设置正态分布的标准差（单位与坐标轴一致，可调节）
# sigma = 0.1  # 控制透明度变化的范围

# # 计算透明度范围（从圆心到边缘）
# num_layers = 10  # 渐变层数
# radii = np.linspace(0, max_radius, num_layers)  # 半径从 0 增加到最大值
# alphas = norm.pdf(radii, 0, sigma)  # 计算透明度（正态分布）
# alphas = np.clip(alphas, 0, 1)  # 确保透明度在 [0,1] 范围内

# # 绘制起点和终点（颜色映射相同，使用正态分布渐变）
# for i, trajectory in enumerate(all_trajectories):
#     x, y = trajectory.T
#     color = cm.jet(colors[i])  # 获取颜色映射
    
#     for point_x, point_y in [(x[0], y[0]), (x[-1], y[-1])]:  # 画起点和终点
#         for radius, alpha in zip(radii, alphas):  # 半径增加，透明度按正态分布变化
#             gradient_circle = Circle(
#                 (point_x, point_y),
#                 radius=radius,  # 半径逐渐增大
#                 color=color,
#                 alpha=alpha,  # 透明度按正态分布计算，并截断到 [0,1]
#                 zorder=4
#             )
#             ax4.add_patch(gradient_circle)

# ax4.set_aspect('equal')
# ax4.set_xlim(-square_size_half, square_size_half)
# ax4.set_ylim(-square_size_half, square_size_half)
# ax4.set_xticks([])
# ax4.set_yticks([])
# ax4.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['bottom'].set_visible(False)
# ax4.spines['left'].set_visible(False)

# # 保存图像
# plt.savefig("brownian_motion_colored_with_normal_gradient_fixed.png", dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()

