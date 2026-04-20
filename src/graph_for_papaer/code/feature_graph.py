import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import scipy.signal
from scipy.signal import butter, lfilter

# --- 全局配置 ---
COLOR_MAP = 'viridis'
C_DARK = "#3057D7"
C_BRIGHT = "#3057D7"

# ===================== 修复：中文显示 =====================
# 1. 加载字体
font_path_cn = '/data/zcx/fonts/SIMSUN.TTC'
font_path_en = '/data/zcx/fonts/TIMES.TTF'

font_cn = fm.FontProperties(fname=font_path_cn)
font_en = fm.FontProperties(fname=font_path_en)

# 2. 关键：直接设置全局字体（不使用 serif，避免覆盖）
plt.rcParams['font.family'] = [font_en.get_name(), font_cn.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
# ==========================================================

FILE_PATH = "/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz/Cargo/12/12_Cargo-Segment_18.wav"

def plot_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    time = np.arange(len(y)) / sr
    nyquist = 0.5 * sr

    plt.figure(figsize=(20, 10))

    # [1] 原始波形
    ax = plt.subplot(2, 4, 1)
    plt.plot(time, y, color=C_DARK, linewidth=0.8)
    plt.title("(a) 原始波形", fontproperties=font_cn)  # 保险：强制中文
    plt.xlabel("时间 (s)", fontproperties=font_cn)
    plt.xlim(0, time[-1])
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # [2] 全频段 PSD
    ax = plt.subplot(2, 4, 2)
    f_psd, psd = scipy.signal.welch(y, fs=sr, nperseg=4096)
    plt.plot(f_psd, librosa.power_to_db(psd, ref=np.max), color=C_BRIGHT)
    plt.title("功率谱密度", fontproperties=font_cn)
    plt.xlabel("频率 (Hz)", fontproperties=font_cn)
    plt.xlim(0, nyquist)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # [3] LOFAR
    ax = plt.subplot(2, 4, 3)
    D_lofar = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    librosa.display.specshow(librosa.amplitude_to_db(D_lofar, ref=np.max),
                             sr=sr, x_axis='time', y_axis='linear', cmap=COLOR_MAP)
    plt.title("LOFAR谱", fontproperties=font_cn)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # [4] 功率谱
    ax = plt.subplot(2, 4, 4)
    D = np.abs(librosa.stft(y))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                             sr=sr, x_axis='time', y_axis='log', cmap=COLOR_MAP)
    plt.title("功率谱", fontproperties=font_cn)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # [5] DEMON 谱优化版
    ax = plt.subplot(2, 4, 5)
    
    # 1. 带通滤波（提取空化噪声能量集中区域）
    low_b, high_b = 2000, 6000 
    b_b, a_b = butter(4, [low_b/nyquist, high_b/nyquist], btype='band')
    y_bandpassed = lfilter(b_b, a_b, y)
    
    # 2. 绝对值全波检波
    y_abs = np.abs(y_bandpassed)
    
    # 3. 低通滤波（平滑包络）
    low_cut = 100
    b_l, a_l = butter(4, low_cut/nyquist, btype='low')
    y_dem = lfilter(b_l, a_l, y_abs)
    
    # 4. 去直流分量（非常重要！否则 0Hz 峰值会压制图像）
    y_dem = y_dem - np.mean(y_dem)
    
    # 5. 功率谱估计 (Welch法)
    # nperseg 建议设大一点以提高频率分辨率
    f_d, p_d = scipy.signal.welch(y_dem, fs=sr, nperseg=16384) 
    
    mask = (f_d > 0.5) & (f_d <= 50) # 通常看 0.5-50Hz 即可发现轴频
    
    # 归一化并绘图
    p_db = librosa.power_to_db(p_d[mask], ref=np.max)
    plt.plot(f_d[mask], p_db, color=C_DARK)
    
    plt.title("DEMON谱 (解调谱)", fontproperties=font_cn)
    plt.xlabel("频率 (Hz)", fontproperties=font_cn)
    plt.xlim(0, 50) # 舰船轴频和叶频通常在这个范围
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # [6] MFCC
    ax = plt.subplot(2, 4, 6)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', sr=sr, cmap=COLOR_MAP)
    plt.title("MFCC", fontproperties=font_cn)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # [7] Mel
    ax = plt.subplot(2, 4, 7)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max),
                             sr=sr, x_axis='time', y_axis='mel', cmap=COLOR_MAP)
    plt.title("Mel谱图", fontproperties=font_cn)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # [8] CQT
    ax = plt.subplot(2, 4, 8)
    cqt = np.abs(librosa.cqt(y, sr=sr))
    librosa.display.specshow(librosa.amplitude_to_db(cqt, ref=np.max),
                             sr=sr, x_axis='time', y_axis='cqt_note', cmap=COLOR_MAP)
    plt.title("CQT 谱图", fontproperties=font_cn)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.savefig("../graph/underwater_audio_analysis_2rows.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    plot_features(FILE_PATH)