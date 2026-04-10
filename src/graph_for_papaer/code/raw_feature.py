import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# --- 1. 环境配置 ---
plt.rcParams['axes.unicode_minus'] = False 
plt.style.use('seaborn-v0_8-paper')

# 模拟数据路径与参数
CLASS_NAMES = ["Cargo", "Passengership", "Tanker", "Tug"]
FILE_PATH_TEMPLATE = "/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz/{}/13/13_{}-Segment_8.wav"
SR = 16000

# --- 2. 绘图初始化 ---
# 4行 (Waveform, STFT, Mel, CQT) x 4列 (Classes)
fig, axes = plt.subplots(4, 4, figsize=(18, 14), constrained_layout=True)
colors = ["#4C72B0"] * 4 

for i, name in enumerate(CLASS_NAMES):
    # 加载音频
    file_path = FILE_PATH_TEMPLATE.format(name, name)
    y, sr = librosa.load(file_path, sr=SR)
    
    # --- Row 1: Raw Waveform (Time Domain) ---
    ax_wave = axes[0, i]
    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color=colors[i], alpha=0.8)
    ax_wave.set_title(f"{name}\nRaw Waveform", fontsize=14, fontweight='bold')
    if i == 0: ax_wave.set_ylabel("Amplitude")

    # --- Row 2: STFT (Linear Frequency) ---
    ax_stft = axes[1, i]
    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)
    img_stft = librosa.display.specshow(stft_db, sr=sr, hop_length=512, 
                                        x_axis='time', y_axis='linear', 
                                        ax=ax_stft, cmap='magma')
    ax_stft.set_title("STFT (Linear)", fontsize=12)
    if i == 0: ax_stft.set_ylabel("Freq (Hz)")

    # --- Row 3: Mel Spectrogram (Mel Scale) ---
    ax_mel = axes[2, i]
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img_mel = librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512, 
                                       x_axis='time', y_axis='mel', 
                                       ax=ax_mel, cmap='viridis')
    ax_mel.set_title("Mel Spectrogram", fontsize=12)
    if i == 0: ax_mel.set_ylabel("Mel Freq")

    # --- Row 4: CQT (Constant-Q Transform) ---
    ax_cqt = axes[3, i]
    # CQT typically requires a minimum signal length; 3s at 16k is plenty
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512))
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    img_cqt = librosa.display.specshow(cqt_db, sr=sr, hop_length=512, 
                                       x_axis='time', y_axis='cqt_note', 
                                       ax=ax_cqt, cmap='inferno')
    ax_cqt.set_title("CQT (Log Freq)", fontsize=12)
    if i == 0: ax_cqt.set_ylabel("Note")

# --- 3. 后处理 ---
# 为底部的图添加共同的 X 轴标签
for ax in axes[3, :]:
    ax.set_xlabel("Time (s)")

# 保存
plt.savefig("../graph/DeepShip_MultiFeature_Comparison.png", dpi=600, bbox_inches='tight')
plt.show()