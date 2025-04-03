import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import time

# 模拟参数（固定部分）
square_size = 1500  # nm
grid_bin = 50
D1_um = 0.2
time_interval = 0.006
steps = 100
num_mol = 100000
loc_pre = 25 # unit nm

disp_thre = 570  # nm
bin_num = 50
select_size = 1500  # nm
N = 20  # 重复次数

# Numba 加速函数
def generate_slow_flags(slow_region_fraction, grid_bin):
    all_indices = [(i, j) for i in range(grid_bin) for j in range(grid_bin)]
    num_slow = int(slow_region_fraction * len(all_indices))
    chosen_indices = np.random.choice(len(all_indices), size=num_slow, replace=False)
    slow_flags = np.zeros((grid_bin, grid_bin), dtype=np.bool_)
    for idx in chosen_indices:
        i, j = all_indices[idx]
        slow_flags[i, j] = True
    return slow_flags

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

        # disp = np.sqrt((current[0] - start[0]) ** 2 + (current[1] - start[1]) ** 2)

        idx = 2 * m
        
        loc_uncertainty = np.random.normal(0, loc_pre,size=(4))

        xc_all[idx] = start[0]+loc_uncertainty[0]
        yc_all[idx] = start[1]+loc_uncertainty[1]
        
        xc_all[idx + 1] = current[0]+loc_uncertainty[2]
        yc_all[idx + 1] = current[1]+loc_uncertainty[3]
        
        zc_all[idx] = np.sqrt((xc_all[idx + 1]- xc_all[idx]) ** 2 + (yc_all[idx + 1] - yc_all[idx]) ** 2)
        zc_all[idx + 1] = zc_all[idx]

    return xc_all, yc_all, zc_all

def double_twoD_diff_dis_func_no_b(x, a1, c1, a2, c2):
    return 2 * c1 * x / a1 * np.exp(-x**2 / a1) + 2 * c2 * x / a2 * np.exp(-x**2 / a2)

def double_twoD_diff_dis_func(x, a1, c1, a2, c2, bg):
    return 2 * c1 * x / a1 * np.exp(-x**2 / a1) + 2 * c2 * x / a2 * np.exp(-x**2 / a2) + bg*x

# 主循环：参数组合
slow_vals = np.arange(0.4, 0.61, 0.1)
# slow_vals = np.array([0.37])
# D2_vals = np.arange(2.0, 2.1, 0.1)
D2_vals = np.array([2.2])

summary_results = []  # 每行：[slow_region_fraction, D2_um, P_larger, D_larger]
square_half = square_size / 2

start_time = time.time()

for slow_region_fraction in slow_vals:
    for D2_um in D2_vals:
        results_array = np.zeros((N, 4))
        slow_flags = generate_slow_flags(slow_region_fraction, grid_bin)

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
                continue

            hist_y, bin_edges = np.histogram(filtered_disp, bins=bin_num, range=(0, disp_thre))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            try:
                p0 = [30000, num_mol*10, 10000, num_mol*6]
                bounds_lower = [1e-6] * 4
                bounds_upper = [np.inf] * 4
                params, _ = curve_fit(
                    double_twoD_diff_dis_func_no_b,
                    bin_centers, hist_y,
                    p0=p0, bounds=(bounds_lower, bounds_upper)
                )
                a1, c1, a2, c2= params
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
                
                if D_fit_1>D_fit_2:    # 大的扩散速率在前
                    results_array[trial] = [D_fit_1, percent1, D_fit_2, percent2]
                else:
                    results_array[trial] = [D_fit_2, percent2, D_fit_1, percent1]
            except:
                continue

        results_array = results_array[~np.isnan(results_array).any(axis=1)]
        if len(results_array) == 0:
            continue

        avg = np.mean(results_array, axis=0)
        D1, P1, D2, P2 = avg
        std = np.std(results_array, axis=0)
        D1_std,P1_std,D2_std,P2_std = std
        
        summary_results.append([slow_region_fraction, D2_um, P1, D1, D2, P1_std,D1_std,D2_std])
        # summary_results.append([slow_region_fraction, D2_um, P2, D2, D1])

# 转为 NumPy 数组
summary_results = np.array(summary_results)

# 打印结果
print("slow_region_fraction | D2_um | P_larger (%) | D_larger (um^2/s)| D_smaller (um^2/s)")
for row in summary_results:
    print(f"{row[0]:.2f}\t {row[1]:.1f}\t {row[2]:.2f}\t {row[3]:.2f}\t {row[4]:.2f}")
    
print("--- %s seconds ---" % (time.time() - start_time))
