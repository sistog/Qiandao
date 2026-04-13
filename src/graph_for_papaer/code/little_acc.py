import matplotlib.pyplot as plt
import numpy as np

# 1. 设置全局字体和刻度
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 2. 数据准备
categories = ['5%', '10%', '15%', '20%']
models = ["Beats", "MFCA-Net", "ResNet", "CNN", "LSTM"]

plot_results = {
    "Beats": [0.5759, 0.6031, 0.6304, 0.6587],
    "MFCA-Net": [0.5036, 0.5530, 0.5690, 0.5875],
    "ResNet": [0.5304, 0.5656, 0.5851, 0.6137],
    "CNN": [0.5023, 0.5410, 0.5537, 0.5742],
    "LSTM": [0.4878, 0.5037, 0.5325, 0.5567]
}

# 颜色配置 (保持原代码风格)
colors = ['#B22222', '#08573B', '#4682B4', '#DAA520', '#808080']

# 3. 创建 2x2 子图布局
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=120)
axes = axes.flatten() # 将 2x2 阵列展平为长度为 4 的列表

# 4. 循环绘制每个类别的柱状图
for i, cat in enumerate(categories):
    ax = axes[i]
    
    # 提取当前类别下所有模型的值
    values = [plot_results[model][i] for model in models]
    
    # 绘制柱状图
    bars = ax.bar(models, values, color=colors, edgecolor='black', alpha=0.8)
    
    # 设置子图细节
    ax.set_title(f'Train Ration: {cat}', fontsize=16, fontweight='bold')
    ax.set_ylim(0.45, 0.70) # 根据数据范围调整 y 轴
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.grid(axis='y', ls='--', alpha=0.5)
    
    # 在柱子上标注具体数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)

# 5. 整体布局调整
plt.tight_layout()

# 保存并显示
plt.savefig('Accuracy_Bar_Comparison.png', dpi=600)
plt.show()