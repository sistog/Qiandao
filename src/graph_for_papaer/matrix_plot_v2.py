import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 准备数据
data = {
    'MFCA-Net': [[1938, 210, 401, 128], [275, 2822, 402, 290], [552, 198, 2605, 232], [132, 214, 515, 2489]],
    'ResNet18': [[1741, 177, 594, 165], [149, 2679, 452, 509], [539, 425, 2606, 17], [133, 561, 128, 2528]],
    'AST': [[1890, 119, 536, 132], [264, 2612, 585, 328], [528, 249, 2793, 17], [333, 717, 235, 2065]],
    'HTS-AT': [[1796, 109, 637, 135], [348, 2523, 606, 312], [506, 342, 2730, 9], [326, 761, 356, 1907]]
}

labels = ['Cargo', 'Passenger', 'Tanker', 'Tug']
Chinese_labels = ['货船', '客船', '油轮', '拖船']
models = list(data.keys())

from matplotlib.font_manager import FontProperties

# ===================== 加载宋体 =====================
simsun = FontProperties(fname='/data/zcx/fonts/SIMSUN.TTC', size=20)
Timenew_Roman = FontProperties(fname='/data/zcx/fonts/TIMES.TTF', size=20)

# 全局配置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 计算全局最大值
all_values = [val for matrix in data.values() for row in matrix for val in row]
v_max = max(all_values)

# MFCA-Net

figure= plt.figure(figsize=(10, 5))

sns.heatmap(
        data['MFCA-Net'], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        vmin=0, 
        vmax=v_max,
        cbar=False,
        square=True, 
        xticklabels=Chinese_labels, 
        yticklabels=Chinese_labels,
        annot_kws={"size": 18}
    )
# plt.title('混淆矩阵',fontproperties=simsun)
plt.xlabel('预测标签', fontsize=20, fontproperties=simsun)
plt.ylabel('真实标签', fontsize=20, fontproperties=simsun)
plt.xticks(fontproperties=simsun, fontsize=20)
plt.yticks(fontproperties=simsun, fontsize=20)
plt.tight_layout(pad=2.0)
plt.savefig('MFCA-Net_confusion_matrix.png', bbox_inches='tight', dpi=600)
plt.close()


# 3. 创建子图
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

for i, model_name in enumerate(models):
    ax = axes.flat[i]
    sns.heatmap(
        data[model_name], 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        vmin=0, 
        vmax=v_max,
        cbar=False,
        square=True, 
        xticklabels=Chinese_labels, 
        yticklabels=Chinese_labels,
        ax=ax,
        annot_kws={"size": 18}
    )

    # ===================== 所有中文都指定宋体 =====================
    ax.set_title(f'({chr(97+i)}) {model_name}', fontsize=20, pad=20, fontweight='bold', fontproperties=Timenew_Roman)
    
    # X 轴标签（预测标签）
    ax.set_xticklabels(Chinese_labels, fontproperties=simsun, fontsize=20)
    # Y 轴标签（真实标签）
    ax.set_yticklabels(Chinese_labels, fontproperties=simsun, fontsize=20)

    if i >= 2:
        ax.set_xlabel('预测标签', fontsize=20, fontproperties=simsun)
    if i % 2 == 0:
        ax.set_ylabel('真实标签', fontsize=20, fontproperties=simsun)

plt.tight_layout(pad=2.0)

# 服务器必须用 savefig，不能用 show()
plt.savefig('vessel_confusion_matrix_comparison.png', bbox_inches='tight', dpi=600)
plt.close()

print("混淆矩阵已保存！")