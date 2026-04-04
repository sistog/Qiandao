import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from PIL import Image

# 创建一个简单的图
G = nx.erdos_renyi_graph(9, 0.5)

# 创建图像
fig, ax = plt.subplots(figsize=(8, 8))

# 获取节点的位置
pos = nx.spring_layout(G)

# 绘制网络结构
nx.draw(G, pos, ax=ax, with_labels=True, node_size=500, node_color='cyan', edge_color='black')

# 为每个节点添加梅尔频谱图图像块
for i, (x, y) in pos.items():
    # 创建一个简单的梅尔频谱图图像作为示例
    img_data = np.random.rand(10, 10)  # 这里使用随机数据代替梅尔频谱图数据
    axin = fig.add_axes([x, y, 0.1, 0.1])  # 设置每个图像块的位置
    axin.imshow(img_data, cmap=cm.viridis)
    axin.axis('off')

# 绘制外部的虚线框
box_x = [0.6, 0.8]  # 设定虚线框的位置（x轴）
box_y = [0.6, 0.8]  # 设定虚线框的位置（y轴）
ax.plot(box_x, [box_y[0], box_y[0]], linestyle='--', color='black', lw=2)  # 虚线的上边框
ax.plot(box_x, [box_y[1], box_y[1]], linestyle='--', color='black', lw=2)  # 虚线的下边框
ax.plot([box_x[0], box_x[0]], box_y, linestyle='--', color='black', lw=2)  # 虚线的左边框
ax.plot([box_x[1], box_x[1]], box_y, linestyle='--', color='black', lw=2)  # 虚线的右边框


ax.plot([1, 1], [0, 1], linestyle='--', color='black', lw=2)  # 虚线的右边框

plt.savefig("assets/feature_visualization.png")  # 保存图像到文件

# 显示图像
plt.show()