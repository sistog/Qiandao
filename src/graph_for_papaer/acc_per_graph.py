import matplotlib.pyplot as plt
import numpy as np

# 设置全局刻度字体大小
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# 1. 汇总五个模型的数据
data = {
    # 'Beats': [[1885, 201, 465, 126], [140, 2858, 410, 381], [368, 341, 2855, 23], [144, 513, 97, 2596]],
    'MFCA-Net': [[1938, 210, 401, 128], [275, 2822, 402, 290], [552, 198, 2605, 232], [132, 214, 515, 2489]],
    'ResNet': [[1741, 177, 594, 165], [149, 2679, 452, 509], [539, 425, 2606, 17], [133, 561, 128, 2528]],
    'CNN': [[1890, 119, 536, 132], [264, 2612, 585, 328], [528, 249, 2793, 17], [333, 717, 235, 2065]],
    'LSTM': [[1796, 109, 637, 135], [348, 2523, 606, 312], [506, 342, 2730, 9], [326, 761, 356, 1907]]
}

categories = ['Cargo', 'Passenger', 'Tanker', 'Tug', 'Mean']

# 2. 计算绘图数据
plot_results = {}
for name, matrix in data.items():
    cm = np.array(matrix)
    per_class_acc = np.diag(cm) / np.sum(cm, axis=1) # 召回率/类别准确率
    overall_acc = np.sum(np.diag(cm)) / np.sum(cm)
    plot_results[name] = np.append(per_class_acc, overall_acc)

# 3. 绘图设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

styles = {
    # 'Beats': {'color': '#B22222', 'marker': 'o', 'ls': '-', 'lw': 2.5}, # 突出显示
    'MFCA-Net': {'color': '#B22222', 'marker': 'D', 'ls': '-', 'lw': 2.5},      # 绿色虚线
    'ResNet': {'color': '#4682B4', 'marker': 's', 'ls': '-.', 'lw': 1.2},
    'CNN': {'color': '#DAA520', 'marker': '^', 'ls': ':', 'lw': 1.2},
    'LSTM': {'color': '#808080', 'marker': 'x', 'ls': '--', 'lw': 1.0}
}

fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

for name, vals in plot_results.items():
    ax.plot(categories, vals, **styles[name], label=name)

# 4. 美化
ax.set_ylabel('Recognition Accuracy', fontsize=18)
ax.set_xlabel('Vessel Category', fontsize=18)
ax.set_ylim(0.55, 0.8)

ax.grid(axis='y', ls='--', alpha=0.5)
ax.legend(frameon=True, loc='lower left', fontsize=14)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300) # 保存为高分辨率 PNG 文件
plt.show()