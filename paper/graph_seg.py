import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 读取图像
img = Image.open("assets/mel_spectrogram.png").resize((300, 300))

patch_size = 100

fig = plt.figure(figsize=(12, 6))

# ===== 上面 3×3 =====
for i in range(9):
    ax = fig.add_subplot(4, 9, (i//3)*3 + (i%3) + 1)
    
    x = (i % 3) * patch_size
    y = (i // 3) * patch_size
    
    patch = img.crop((x, y, x + patch_size, y + patch_size))
    
    ax.imshow(patch)
    ax.axis("off")

# ===== 下面 1×9 =====
for i in range(9):
    ax = fig.add_subplot(4, 9, 28 + i)  # 第4行
    
    x = (i % 3) * patch_size
    y = (i // 3) * patch_size
    
    patch = img.crop((x, y, x + patch_size, y + patch_size))
    
    ax.imshow(patch)
    ax.axis("off")

plt.tight_layout()
plt.savefig("assets/mel_spectrogram_parts.png")  # 保存图像到文件
plt.show()