import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# ===================== 服务器专用：加载宋体（绝对路径）=====================
# 直接加载你提供的 SIMSUN.TTC，不依赖系统字体
simsun = FontProperties(fname='/data/zcx/fonts/SIMSUN.TTC', size=28)

# 全局配置（英文用 Times New Roman，中文用你路径里的宋体）
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24

# 2. 数据准备
categories = ['5%', '10%', '15%', '20%']
models = ["BEATs", "MFCA-Net", "ResNet18", "AST", "HTS-AT"]

plot_results = {
    "BEATs": [0.5759, 0.6031, 0.6304, 0.6587],
    "MFCA-Net": [0.5036, 0.5530, 0.5690, 0.5875],
    "ResNet18": [0.5304, 0.5656, 0.5851, 0.6137],
    "AST": [0.5023, 0.5410, 0.5537, 0.5742],
    "HTS-AT": [0.4878, 0.5037, 0.5325, 0.5567]
}

# 颜色配置
colors = ["#C99191C2", "#8CDEC2", "#9FC8EB", "#E6D6AD", "#BEBABA"]

# 3. 创建 2x2 子图布局
fig, axes = plt.subplots(2, 2, figsize=(19, 14), dpi=120)
axes = axes.flatten()

# 4. 循环绘制每个类别的柱状图
for i, cat in enumerate(categories):
    ax = axes[i]
    
    values = [plot_results[model][i] for model in models]
    bars = ax.bar(models, values, color=colors, edgecolor='black', alpha=0.8)
    
    # ===================== 所有中文都指定宋体 =====================
    ax.set_title(f'训练集比例: {cat}', fontsize=28, fontweight='bold', fontproperties=simsun)
    ax.set_ylabel('准确率', fontsize=28, fontproperties=simsun)
    ax.grid(axis='y', ls='--', alpha=0.5)
    ax.set_ylim(0.4, 0.7)  # 设置 y 轴范围
    
    # 柱子数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=24)

# 整体布局
plt.tight_layout()

# 保存（服务器必须用 savefig，不能用 plt.show()）
plt.savefig('Accuracy_Bar_Comparison.png', dpi=600, bbox_inches='tight')
plt.close()  # 释放内存