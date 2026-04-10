import numpy as np
import matplotlib.pyplot as plt

# 统一配色方案
COLOR_LINE_SPECTRUM = "#336fe7"  # 桃红色，用于线谱竖线
COLOR_CONTINUOUS_ENV = "#336fe7" # 深紫色，用于连续谱包络
COLOR_STEM_HEAD = "#336fe7"     # 极深紫色，用于火柴头圆点

# 中文及负号显示配置
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

def plot_simulated_spectrums():
    # 1. 模拟数据生成
    f = np.linspace(5, 95, 200) 
    n_f_continuous = -0.003 * (f - 50)**2 + 10  # 连续谱抛物线
    
    f_line = np.array([20, 40, 60, 80]) # 线谱位置
    n_f_line_base = -0.003 * (f_line - 50)**2 + 10
    n_f_line = n_f_line_base * (1 + 1.2 * np.random.rand(4)) # 加上随机高度

    plt.figure(figsize=(15, 5))

    # --- (a) 连续谱 ---
    plt.subplot(1, 3, 1)
    plt.plot(f, n_f_continuous, color=COLOR_CONTINUOUS_ENV, linewidth=2)
    plt.title("(a) 连续谱")
    plt.xlabel("频率 (f)")
    plt.ylabel("幅度 n(f)")
    plt.xlim(0, 100)
    plt.ylim(0, np.max(n_f_line) + 2)
    plt.grid(False)

    # --- (b) 线谱 ---
    plt.subplot(1, 3, 2)
    # 关键修改：linefmt 仅设置线型 '-'，颜色通过 setp 设置
    markerline, stemlines, baseline = plt.stem(f_line, n_f_line, linefmt='-', markerfmt='o', basefmt='k-')
    
    # 使用 setp 精细控制颜色
    plt.setp(stemlines, color=COLOR_LINE_SPECTRUM, linewidth=1.5)
    plt.setp(markerline, markerfacecolor=COLOR_STEM_HEAD, markeredgecolor=COLOR_STEM_HEAD, markersize=5)
    
    plt.title("(b) 线谱")
    plt.xlabel("频率 (f)")
    plt.xlim(0, 100)
    plt.ylim(0, np.max(n_f_line) + 2)
    plt.grid(False)

    # --- (c) 混合谱 ---
    plt.subplot(1, 3, 3)
    # 绘制底层连续谱
    plt.plot(f, n_f_continuous, color=COLOR_CONTINUOUS_ENV, linewidth=1.5, alpha=0.6)
    
    background_at_lines = -0.003 * (f_line - 50)**2 + 10

    # 3. 将线谱原有的增益高度加到背景高度上
    n_f_line_combined = n_f_line + background_at_lines
    # 绘制叠加线谱
    markerline, stemlines, baseline = plt.stem(f_line, n_f_line_combined, linefmt='-', markerfmt='o', basefmt='k-')
    plt.setp(stemlines, color=COLOR_LINE_SPECTRUM, linewidth=1.5)
    plt.setp(markerline, markerfacecolor=COLOR_STEM_HEAD, markeredgecolor=COLOR_STEM_HEAD, markersize=5)
    
    plt.title("(c) 线谱和连续谱")
    plt.xlabel("频率 (f)")
    plt.xlim(0, 100)
    plt.ylim(0, np.max(n_f_line) + 2)
    plt.grid(False)

    plt.tight_layout()
    plt.savefig("simulated_spectrums.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    plot_simulated_spectrums()