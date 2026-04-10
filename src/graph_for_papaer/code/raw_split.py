import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display


FILE_PATH = "/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz/Cargo/5/5_Cargo-Segment_12.wav"

waveform, sr = librosa.load(FILE_PATH, sr=None)
tmp = waveform[:int(32000*2.5)]  # 取前2.5秒
duration = len(tmp) / sr
t = np.linspace(0, duration, len(tmp))
fs = sr
tmp = (waveform - waveform.mean())*8
audio_track = tmp[:int(32000*2.5)]

# # 1. 生成模拟音频数据
# fs = 1000  # 采样率
# t = np.linspace(0, 10, fs * 10)  # 10秒数据
# # 创建一个有起伏的随机波形
# audio_track = np.sin(2 * np.pi * 0.5 * t) * 0.5 + np.random.normal(0, 0.1, len(t))
# audio_track = audio_track * np.hanning(len(t)) # 让边缘圆滑一点

# 定义分割点 (单位: 秒)
segments = [(0.1, 0.4), (1.2, 1.5), (2.1, 2.4)]

# 2. 创建画布
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.4)

# --- 顶部：原始音频与分割标记 ---
ax0 = fig.add_subplot(gs[0])
ax0.fill_between(t, audio_track, color='#2c7bb6', alpha=0.9)
# ax0.set_title("Audio Track", fontsize=16, pad=-30, fontweight='bold')
ax0.set_xlabel("Segmentation", color='#e66101', fontsize=18, fontweight='bold', labelpad=15)

# 绘制橙色矩形框
for start, end in segments:
    rect = patches.Rectangle((start, -1), end-start, 2, 
                             linewidth=2, edgecolor='#e66101', facecolor='none', zorder=3)
    ax0.add_patch(rect)

# ax0.set_ylim(-, 1.2)
ax0.axis('off') # 隐藏坐标轴以贴合原图风格

# --- 底部：分割后的片段 ---
# 我们使用一个子图，通过调整 x 轴位置来模拟“独立块”的效果
ax1 = fig.add_subplot(gs[1])
spacing = 0.5 # 块之间的间距
current_x = 0

for start, end in segments:
    # 提取对应数据
    mask = (t >= start) & (t <= end)
    seg_t = t[mask]
    seg_data = audio_track[mask]
    
    # 归一化时间轴以便并排显示
    display_t = (seg_t - start) + current_x
    dur = end - start
    
    # 修复点：使用 FancyBboxPatch 代替 Rectangle
    rect = patches.FancyBboxPatch(
        (current_x, -1), dur, 2, 
        color='white', 
        ec='#ccc', 
        lw=1, 
        zorder=1, 
        boxstyle="round,pad=0.05"  # 现在这个参数合法了
    )
    ax1.add_patch(rect)
    # 绘制波形
    ax1.fill_between(display_t, seg_data, color='#2c7bb6', zorder=2)
    
    current_x += dur + spacing

# 绘制中间的虚线
ax1.axhline(0, color='#2c7bb6', linestyle=':', lw=1.5, zorder=0)

# ax1.set_ylim(-1.2, 1.2)
ax1.set_xlim(-0.2, current_x)
ax1.axis('off')

plt.savefig("../graph/raw_split_visualization.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()