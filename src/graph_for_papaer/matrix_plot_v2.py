import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 准备数据 (基于 2026-04 实验结果)
# 类别顺序: Cargo, Passenger, Tanker, Tug
data = {
    'Transformer': [[1938, 210, 401, 128], [275, 2822, 402, 290], [552, 198, 2605, 232], [132, 214, 515, 2489]],
    'ResNet': [[1741, 177, 594, 165], [149, 2679, 452, 509], [539, 425, 2606, 17], [133, 561, 128, 2528]],
    'CNN': [[1890, 119, 536, 132], [264, 2612, 585, 328], [528, 249, 2793, 17], [333, 717, 235, 2065]],
    'LSTM': [[1796, 109, 637, 135], [348, 2523, 606, 312], [506, 342, 2730, 9], [326, 761, 356, 1907]]
}

labels = ['Cargo', 'Passenger', 'Tanker', 'Tug']
models = list(data.keys())

# 2. 设置全局绘图参数 (学术顶刊风格)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 计算全局最大值以统一颜色标尺
all_values = [val for matrix in data.values() for row in matrix for val in row]
v_max = max(all_values)

# 3. 创建 2x2 子图布局
fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)

for i, model_name in enumerate(models):
    ax = axes.flat[i]
    sns.heatmap(data[model_name], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                vmin=0, 
                vmax=v_max,  # 核心：统一颜色映射范围
                cbar=False,  # 只在右上角子图保留颜色条，减少冗余
                square=True, 
                xticklabels=labels, 
                yticklabels=labels,
                ax=ax,
                annot_kws={"size": 11},
                cbar_kws={'label': 'Count' if i == 1 else ''})
    
    ax.set_title(f'({chr(97+i)}) {model_name}', fontsize=14, pad=12, fontweight='bold')
    
    # 移除中间子图的冗余标签
    if i >= 2: ax.set_xlabel('Predicted Label', fontsize=12)
    if i % 2 == 0: ax.set_ylabel('True Label', fontsize=12)

# 4. 极致紧凑化处理
plt.tight_layout(pad=2.0)

# 5. 导出高分辨率 PDF (矢量图，适合打印)
plt.savefig('vessel_confusion_matrix_comparison.png', bbox_inches='tight', dpi=600)
plt.show()

print("优化后的混淆矩阵已保存为 vessel_confusion_matrix_comparison.png")