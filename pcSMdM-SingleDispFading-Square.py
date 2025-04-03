# 由于代码执行状态被重置，需要重新加载所有库并重新生成数据

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.cm as cm
from scipy.stats import norm
from PIL import Image


# ================= 参数设置 =================
pixel_size = 114  # unit is nm
slow_rangenm = 0  # unit is nm

D1_um = 2  # unit is um2/s
D2_um = 2  # unit is um2/s
time_interval = 0.005  # unit is second
num_molecule = 500 # 分子数量

D1 = D1_um * 10**6 / pixel_size / pixel_size  # 中心区域扩散系数
D2 = D2_um * 10**6 / pixel_size / pixel_size  # 外围区域扩散系数
steps = 1  # 模拟步数
t = time_interval / steps  # 单步时间

slow_range = slow_rangenm / pixel_size / 2
square_size = 7.0175438596491228070175438596491  # size of simulation
square_size_half = square_size / 2

# 设定颜色映射的范围
lower_limit = 0  # 设置下限
upper_limit = 4   # 设置上限

# 固定随机种子，确保所有图的点的坐标一致
np.random.seed(42)

# ================= 初始化坐标 =================
start_points = np.random.uniform(-square_size_half, square_size_half, (num_molecule, 2))
# start_points = [[0,0]]

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

# 重新模拟布朗运动，保证所有图的点坐标一致
all_data = [simulate_brownian_motion(start_point) for start_point in start_points]
all_trajectories = [data[0] for data in all_data]
displacements = np.array([data[1] for data in all_data])  # 计算位移

# 归一化位移用于颜色映射
displacements = np.clip(displacements, lower_limit, upper_limit)
colors = (displacements - lower_limit) / (upper_limit - lower_limit)




# 自定义高斯函数，使中心值为1，远处趋近于0
def gaussian_alpha(distance, sigma):
    """
    计算基于高斯函数的透明度

    :param distance: 点到圆心的距离
    :param sigma: 控制透明度衰减的标准差，单位与坐标轴一致
    :return: 透明度值，范围在 (0,1] 之间
    """
    return np.exp(- (distance**2) / (2 * sigma**2))

# ================= 重新生成第四张图（使用自定义高斯函数） =================

fig, ax4 = plt.subplots(figsize=(square_size, square_size))

# 设置圆的最大半径（单位与坐标轴一致，可调节）
max_radius = 0.4  # 设定最大圆半径

# 设置高斯函数的标准差（单位与坐标轴一致，可调节）
sigma = 0.1  # 控制透明度变化的范围

# 计算透明度范围（从圆心到边缘）
num_layers = 50  # 渐变层数
radii = np.linspace(0, max_radius, num_layers)  # 半径从 0 增加到最大值
alphas = gaussian_alpha(radii, sigma)*0.5  # 计算透明度（使用自定义高斯函数）

# 绘制起点和终点（颜色映射相同，使用高斯渐变透明度）
for i, trajectory in enumerate(all_trajectories):
    x, y = trajectory.T
    color = cm.jet(colors[i])  # 获取颜色映射

    for point_x, point_y in [(x[0], y[0]), (x[-1], y[-1])]:  # 画起点和终点
        for radius, alpha in zip(radii, alphas):  # 半径增加，透明度按高斯函数变化
            gradient_circle = Circle(
                (point_x, point_y),
                radius=radius,  # 半径逐渐增大
                color=color,
                alpha=alpha,  # 透明度按自定义高斯函数计算
                zorder=4
            )
            ax4.add_patch(gradient_circle)

ax4.set_aspect('equal')
ax4.set_xlim(-square_size_half, square_size_half)
ax4.set_ylim(-square_size_half, square_size_half)
ax4.set_xticks([])
ax4.set_yticks([])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)

# 保存图像
plt.savefig("brownian_motion_colored_with_custom_gaussian.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()



