import matplotlib.pyplot as plt
import numpy as np

# 1. 准备您的真实数据
# 这是一个示例。您需要将您的真实值填入这些列表。
# 确保数据点数量与 vessels 的数量匹配 (5 个)。
methods = ['Cargo', 'Passenger Ship', 'Tanker', 'Tug', 'Mean']
vessels = np.arange(len(methods)) # 生成 [0, 1, 2, 3, 4] 用于放置

# 用您的真实数据替换这些占位符 [v1, v2, v3, v4, v5]
# 确保每个列表恰好有 5 个值。
# 注意：v1 到 v5 应为浮点数，如 0.70, 0.57, 0.95, 0.71, 0.73 等。
y_scae =        [0.70, 0.57, 0.95, 0.71, 0.73]
y_uart =        [0.49, 0.78, 0.92, 0.77, 0.75]
y_agnet =       [0.62, 0.74, 0.92, 0.72, 0.75]
y_snanet =      [0.66, 0.73, 0.89, 0.76, 0.76]
y_joint_model = [0.66, 0.76, 0.87, 0.78, 0.76]
y_ualf =        [0.74, 0.77, 0.89, 0.77, 0.79]
y_sfc_sup =     [0.79, 0.81, 0.91, 0.77, 0.82] # 您的主要方法

# 2. 设置绘图
fig, ax = plt.subplots(figsize=(8, 6), dpi=100) # 创建图和轴，dpi 设为 100 以获得更高的分辨率

# 3. 绘制现有（基线）模型 - 全部使用虚线
# 为基线方法使用 'dashed' (或 '--') 线型。
# 为每个标记使用 'v' (下三角), 'o' (圆圈), 's' (正方形), '^' (上三角), 'x' (X 形), '*' (星形)。
ax.plot(vessels, y_scae, marker='o', linestyle='dashed', label='SCAE', color='#0077c8') # 蓝色
ax.plot(vessels, y_uart, marker='^', linestyle='dashed', label='UART', color='#e05a00') # 橙色
ax.plot(vessels, y_agnet, marker='s', linestyle='dashed', label='AGNet', color='#f0a30a') # 黄色
ax.plot(vessels, y_snanet, marker='v', linestyle='dashed', label='SNANet', color='#903090') # 紫色
ax.plot(vessels, y_joint_model, marker='*', linestyle='dashed', label='Joint Model', color='#70a030') # 绿色
ax.plot(vessels, y_ualf, marker='x', linestyle='dashed', label='UALF', color='#40b0f0') # 浅蓝色

# 4. 绘制您的主要建议方法 - 实线且加粗
# 为主要方法使用 'solid' (或 '-') 线型，并将 linewidth 设为 2.5 或 3.0。
ax.plot(vessels, y_sfc_sup, marker='o', linestyle='solid', label='SFC-Sup(proposed)', color='#a01020', linewidth=3.0) # 深红色

# 5. 轴标签和标题
ax.set_xlabel('The Type of Vessel', fontname='Times New Roman', fontsize=12)
ax.set_ylabel('Average Recognition Accuracy', fontname='Times New Roman', fontsize=12)

# 6. 设置 X 轴刻度标签和位置
ax.set_xticks(vessels) # 在 0, 1, 2, 3, 4 处设置刻度
ax.set_xticklabels(methods, fontname='Times New Roman', fontsize=10) # 标签

# 7. 格式化 Y 轴（例如：显示 .5 到 .9）
ax.set_yticks(np.arange(0.5, 1.0, 0.1)) # 设置 .5 到 .9，步长 .1
ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0.5, 1.0, 0.1)], fontname='Times New Roman', fontsize=10)

# 8. 图例
ax.legend(frameon=True, edgecolor='black') # 手动设置字体和边框

# 9. 格式化图表边框和外观
# 设为白色背景
fig.patch.set_facecolor('white')
# 移除所有脊（可选，如果您想要无边框的外观。示例图保留了边框，所以我们保持默认设置。）

plt.tight_layout() # 调整间距以避免标签被切断
plt.savefig('example_plot.png', dpi=300) # 保存为高分辨率 PNG 文件
plt.show()