import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import butter, lfilter

# --- 全局配置 ---
COLOR_MAP = 'viridis' # 选择一个适合科学可视化的色图
# 提取 magma 色条中的代表色用于一维曲线
C_DARK = "#3057D7"   # 深紫色 (对应低能量)
C_BRIGHT = "#3057D7" # 桃红色 (对应高能量)

plt.rcParams['axes.unicode_minus'] = False 
plt.style.use('seaborn-v0_8-whitegrid') # 使用更现代的网格风格

FILE_PATH = "/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz/Cargo/12/12_Cargo-Segment_18.wav"

def plot_features(file_path):
    # 1. 数据准备
    y, sr = librosa.load(file_path, sr=16000)
    time = np.arange(len(y)) / sr
    nyquist = 0.5 * sr

    plt.figure(figsize=(20, 10)) # 两行布局建议拉宽

    # --- 第一行：基础时频分析 ---

    # [1] 原始波形
    plt.subplot(2, 4, 1)
    plt.plot(time, y, color=C_DARK, linewidth=0.8)
    plt.title("(a) Original Waveform")
    plt.xlabel("Time (s)")
    plt.xlim(0, time[-1])

    # [2] 全频段 PSD (Welch)
    plt.subplot(2, 4, 2)
    f_psd, psd = scipy.signal.welch(y, fs=sr, nperseg=4096)
    plt.plot(f_psd, librosa.power_to_db(psd, ref=np.max), color=C_BRIGHT)
    plt.title("Full PSD (Welch)")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, nyquist)

    # [3] LOFAR图 (线性频率)
    plt.subplot(2, 4, 3)
    D_lofar = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    librosa.display.specshow(librosa.amplitude_to_db(D_lofar, ref=np.max), 
                             sr=sr, x_axis='time', y_axis='linear', cmap=COLOR_MAP)
    plt.title("LOFAR (Linear)")

    # [4] 功率谱图 (Log频率)
    plt.subplot(2, 4, 4)
    D = np.abs(librosa.stft(y))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), 
                             sr=sr, x_axis='time', y_axis='log', cmap=COLOR_MAP)
    plt.title("Spectrogram (Log)")

    # --- 第二行：深度特征与解调 ---

    # [5] DEMON 谱 (0-100Hz 一维)
    plt.subplot(2, 4, 5)
    b_b, a_b = butter(4, [1000/nyquist, 5000/nyquist], btype='band')
    b_l, a_l = butter(4, 100/nyquist, btype='low')
    y_filtered = lfilter(b_b, a_b, y)
    y_dem = lfilter(b_l, a_l, np.abs(y_filtered))
    f_d, p_d = scipy.signal.welch(y_dem, fs=sr, nperseg=8192)
    mask = (f_d >= 0) & (f_d <= 100)
    plt.plot(f_d[mask], librosa.power_to_db(p_d[mask], ref=np.max), color=C_DARK)
    plt.title("DEMON Spectrum (0-100Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.xlim(0, 100)

    # [6] MFCC
    plt.subplot(2, 4, 6)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # y_axis='mel' 配合 sr 会显示频率刻度，若想显示系数索引可设为 None
    librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', sr=sr, cmap=COLOR_MAP)
    plt.title("MFCCs")

    # [7] Mel 谱图
    plt.subplot(2, 4, 7)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), 
                             sr=sr, x_axis='time', y_axis='mel', cmap=COLOR_MAP)
    plt.title("Mel Spectrogram")

    # [8] CQT 谱图
    plt.subplot(2, 4, 8)
    cqt = np.abs(librosa.cqt(y, sr=sr))
    librosa.display.specshow(librosa.amplitude_to_db(cqt, ref=np.max), 
                             sr=sr, x_axis='time', y_axis='cqt_note', cmap=COLOR_MAP)
    plt.title("CQT Spectrogram")

    plt.tight_layout()
    plt.savefig("../graph/underwater_audio_analysis_2rows.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    plot_features(FILE_PATH)