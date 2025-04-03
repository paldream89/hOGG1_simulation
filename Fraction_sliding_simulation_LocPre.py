import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit, prange
from scipy.optimize import curve_fit
from scipy.integrate import simpson

# 模拟参数
slow_region_fraction = 0.4
D2_um = 1.7

square_size = 1500  # nm
grid_bin = 15
D1_um = 0.2
time_interval = 0.006
steps = 100
num_mol = 10000

disp_thre = 570  # nm
bin_num = 50
select_size = 900  # nm

N = 1 # 重复N次求平均

# 初始化起始点
square_half = square_size / 2
starts = np.random.uniform(-square_half, square_half, size=(num_mol, 2))

# 修改为：将 slow 区域索引提前生成，并传递给模拟函数

# 用 numpy 生成 slow_flags（固定不变）
all_indices = [(i, j) for i in range(grid_bin) for j in range(grid_bin)]
num_slow = int(slow_region_fraction * len(all_indices))
chosen_indices = np.random.choice(len(all_indices), size=num_slow, replace=False)

slow_flags = np.zeros((grid_bin, grid_bin), dtype=np.bool_)
for idx in chosen_indices:
    i, j = all_indices[idx]
    slow_flags[i, j] = True

# 修改 Numba 加速函数，接收 slow_flags 参数
@njit(parallel=True)
def simulate_all_trajectories_fixed_slowflags(
    starts, slow_flags, square_size, grid_bin, D1_um, D2_um, time_interval, steps
):
    num_mol = starts.shape[0]
    total_points = num_mol * 2

    D1 = D1_um * 1e6
    D2 = D2_um * 1e6
    t = time_interval / steps
    square_size_half = square_size / 2
    cell_size = square_size / grid_bin

    xc_all = np.empty(total_points, dtype=np.float32)
    yc_all = np.empty(total_points, dtype=np.float32)
    zc_all = np.empty(total_points, dtype=np.float32)

    for m in prange(num_mol):
        start = starts[m]
        current = start.copy()

        for _ in range(steps):
            x_shifted = current[0] + square_size_half
            y_shifted = current[1] + square_size_half
            i = int(x_shifted // cell_size)
            j = int(y_shifted // cell_size)
            in_slow = False
            if 0 <= i < grid_bin and 0 <= j < grid_bin:
                in_slow = slow_flags[i, j]

            D = D1 if in_slow else D2
            sigma = np.sqrt(2 * D * t)
            dx = np.random.normal(0, sigma)
            dy = np.random.normal(0, sigma)
            current[0] += dx
            current[1] += dy

        disp = np.sqrt((current[0] - start[0]) ** 2 + (current[1] - start[1]) ** 2)

        idx = 2 * m
        xc_all[idx] = start[0]
        yc_all[idx] = start[1]
        zc_all[idx] = disp
        xc_all[idx + 1] = current[0]
        yc_all[idx + 1] = current[1]
        zc_all[idx + 1] = disp

    return xc_all, yc_all, zc_all

# 定义双扩散分布模型拟合函数
def double_twoD_diff_dis_func_no_b(x, a1, c1, a2, c2):
    return 2 * c1 * x / a1 * np.exp(-x**2 / a1) + 2 * c2 * x / a2 * np.exp(-x**2 / a2)

def double_twoD_diff_dis_func(x, a1, c1, a2, c2, bg):
    return 2 * c1 * x / a1 * np.exp(-x**2 / a1) + 2 * c2 * x / a2 * np.exp(-x**2 / a2)+bg*x

# 完整修复并增强版本：重复模拟 + 拟合分析 + 参数约束 + 统计平均

results_array = np.zeros((N, 4))  # 每行对应 [D_fit_1, percent1, D_fit_2, percent2]

# 预生成 slow_flags 一次
all_indices = [(i, j) for i in range(grid_bin) for j in range(grid_bin)]
num_slow = int(slow_region_fraction * len(all_indices))
chosen_indices = np.random.choice(len(all_indices), size=num_slow, replace=False)

# 参数约束（全部参数 > 0）
bounds_lower = [1e-6, 1e-6, 1e-6, 1e-6]
bounds_upper = [np.inf, np.inf, np.inf, np.inf]

# 执行多轮模拟
for trial in range(N):
    starts = np.random.uniform(-square_half, square_half, size=(num_mol, 2))
    xc_all, yc_all, zc_all = simulate_all_trajectories_fixed_slowflags(
        starts, slow_flags, square_size, grid_bin,
        D1_um, D2_um, time_interval, steps
    )

    half_select = select_size / 2
    mask = (
        (zc_all < disp_thre) &
        (xc_all > -half_select) & (xc_all < half_select) &
        (yc_all > -half_select) & (yc_all < half_select)
    )
    filtered_disp = zc_all[mask]

    if len(filtered_disp) < 100:
        continue  # 样本太小跳过拟合

    # 拟合准备
    hist_y, bin_edges = np.histogram(filtered_disp, bins=bin_num, range=(0, disp_thre))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    try:
        p0 = [10000, 10000, 50000, 10000]
        params, _ = curve_fit(
            double_twoD_diff_dis_func_no_b,
            bin_centers, hist_y,
            p0=p0, bounds=(bounds_lower, bounds_upper)
        )
        a1, c1, a2, c2 = params
        D_fit_1 = a1 / 4 / time_interval / 1e6
        D_fit_2 = a2 / 4 / time_interval / 1e6

        x_range = np.linspace(0, disp_thre, 1000)
        comp1 = 2 * c1 * x_range / a1 * np.exp(-x_range**2 / a1)
        comp2 = 2 * c2 * x_range / a2 * np.exp(-x_range**2 / a2)
        area1 = simpson(y=comp1, x=x_range)
        area2 = simpson(y=comp2, x=x_range)
        total_area = area1 + area2
        percent1 = 100 * area1 / total_area
        percent2 = 100 * area2 / total_area

        results_array[trial] = [D_fit_1, percent1, D_fit_2, percent2]
    except Exception:
        results_array[trial] = [np.nan] * 4
        
    # 生成拟合曲线
    x_fit = np.linspace(0, disp_thre, 500)
    y_fit = double_twoD_diff_dis_func_no_b(x_fit, *params)

    # # 绘图
    def comp1_plot(x): 
        return 2 * c1 * x / a1 * np.exp(-x**2 / a1)
    def comp2_plot(x): 
        return 2 * c2 * x / a2 * np.exp(-x**2 / a2)
    def bg_plot(x):
        return bg * x
    plt.figure(figsize=(6, 4))
    plt.hist(filtered_disp, bins=50, range=(0, disp_thre), color='grey', edgecolor='black', label='Histogram')
    plt.plot(x_fit, y_fit, 'black', linewidth=2, label='Double Fit')
    # 组成分量
    plt.plot(x_fit, comp1_plot(x_fit), 'b--', linewidth=2, label='Component 1')
    plt.plot(x_fit, comp2_plot(x_fit), 'r--', linewidth=2, label='Component 2')
    # plt.plot(x_fit, bg_plot(x_fit), 'm--', linewidth=2, label='backgroud')
    plt.xlabel("Displacement (nm)")
    plt.ylabel("Frequency")
    plt.title("Double 2D Diffusion Distribution Fit")
    plt.legend()
    plt.tight_layout()
    plt.show()

# 删除无效行
results_array = results_array[~np.isnan(results_array).any(axis=1)]

# 输出平均结果
average_result = np.mean(results_array, axis=0)

# # 打印输出，保留两位小数
# print(f"D_fit_1 = {D_fit_1:.2f}, contribution = {percent1:.2f}%")
# print(f"D_fit_2 = {D_fit_2:.2f}, contribution = {percent2:.2f}%")

if not np.isnan(average_result).any():
    D1 = average_result[0]
    P1 = average_result[1]
    D2 = average_result[2]
    P2 = average_result[3]

    if D1 >= D2:
        D_larger = D1
        P_larger = P1
        print(f"Larger D: {D1:.2f} µm²/s, Percentage: {P1:.2f}%")
    else:
        D_larger = D2
        P_larger = P2
        print(f"Larger D: {D2:.2f} µm²/s, Percentage: {P2:.2f}%")
else:
    print("average_result contains NaN. Cannot determine larger D and corresponding percentage.")

