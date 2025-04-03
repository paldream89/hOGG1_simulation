import numpy as np
import pandas as pd
import wx
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

# 打开 wxPython 文件选择对话框
app = wx.App(False)
frame = wx.Frame(None, -1, 'Select Excel File')
dialog = wx.FileDialog(frame, "Choose an Excel file", wildcard="Excel files (*.xlsx)|*.xlsx",
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

if dialog.ShowModal() == wx.ID_OK:
    file_path = dialog.GetPath()
else:
    file_path = None

dialog.Destroy()
frame.Destroy()

if file_path:
    # 读取 Excel 文件
    df = pd.read_excel(file_path, header=None)
    Fraction = df.iloc[:, 0].astype(float).values
    Fit_Fraction = df.iloc[:, 1].astype(float).values
    Fit_D2 = df.iloc[:, 2].astype(float).values

    # 构建一维插值器（只对 Fraction 插值）
    frac_interp = interp1d(Fraction, Fit_Fraction, kind='linear', fill_value='extrapolate')
    d2_interp = interp1d(Fraction, Fit_D2, kind='linear', fill_value='extrapolate')

    # 目标值
    target_fit_frac = 65.5
    target_fit_d2 = 0.93

    # 目标函数
    def objective(params):
        f = params[0]
        pred_frac = frac_interp(f)
        pred_d2 = d2_interp(f)
        return ((pred_frac - target_fit_frac)/target_fit_frac)**2 + ((pred_d2 - target_fit_d2)/target_fit_d2)**2

    # 优化范围
    bounds = [(min(Fraction), max(Fraction))]

    result = differential_evolution(objective, bounds)

    best_fraction = result.x[0]
    best_cost = result.fun

    print(f"Estimated Fraction: {best_fraction:.4f}")
    print(f"Fitting cost (squared error): {best_cost:.6f}")
else:
    print("No file selected.")
