import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from matplotlib.font_manager import FontProperties

# --- 1. 环境配置 ---
# 设置全局字体（Linux系统请确保路径正确，或参考前文安装Noto Sans CJK）
# font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc') 
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.style.use('seaborn-v0_8-paper')         # 使用学术论文常用的简洁风格

# 模拟数据路径与参数
CLASS_NAMES = ["Cargo", "Passengership", "Tanker", "Tug"]
FILE_PATH_TEMPLATE = "/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz/{}/13/13_{}-Segment_8.wav"

# --- 2. 绘图初始化 ---
# 增加画布高度，预留标题和标签空间
fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#4C72B0"] # 使用高对比度的学术色系

for i, name in enumerate(CLASS_NAMES):
    # 加载音频
    file_path = FILE_PATH_TEMPLATE.format(name, name)
    y, sr = librosa.load(file_path, sr=16000)
    
    # # 模拟信号用于演示（实际运行时请取消上方注释）
    # sr = 16000
    # y = np.random.normal(0, 0.1, sr * 3) 

    # --- 上排：原始波形 (Time Domain) ---
    ax_top = axes[0, i]
    librosa.display.waveshow(y, sr=sr, ax=ax_top, color=colors[i], alpha=0.8)
    ax_top.set_title(f"{name}\nRaw Waveform", fontsize=14, fontweight='bold', pad=10)
    ax_top.set_xlabel("Time (s)", fontsize=10)
    if i == 0:  # 仅第一列显示纵轴标签，避免重复
        ax_top.set_ylabel("Amplitude", fontsize=10)
    ax_top.grid(True, linestyle='--', alpha=0.6)
    # ax_top.set_ylim(-0.5, 0.5) # 统一纵坐标刻度，方便横向对比

    # --- 下排：梅尔频谱 (Mel Spectrogram) ---
    ax_bottom = axes[1, i]
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    img = librosa.display.specshow(mel_spec_db.T, sr=sr, hop_length=512, 
                                   x_axis='mel', y_axis='time', 
                                   ax=ax_bottom, cmap='viridis') # magma/viridis是感知均匀的色标
    
    ax_bottom.set_title("Mel Spectrogram", fontsize=12)
    ax_bottom.set_xlabel("Frequency (Hz)", fontsize=10)
    if i == 0:
        ax_bottom.set_ylabel("Time (s)", fontsize=10)

# --- 3. 后处理 ---
# 在右侧统一添加一个 Colorbar，避免每个子图都带一个导致画面拥挤
# cbar = fig.colorbar(img, ax=axes[1, :], location='right', shrink=0.8, format='%+2.0f dB')
# cbar.ax.set_ylabel('Power (dB)', rotation=-90, va="bottom")

# 保存为矢量图或高分辨率位图
plt.savefig("DeepShip_Features_Optimized.png", dpi=600, bbox_inches='tight')
plt.show()