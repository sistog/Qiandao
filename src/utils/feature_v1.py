import scipy.signal as signal
from scipy.signal import butter, lfilter
import librosa
import numpy as np
from matplotlib import pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置中文字体为 SimSun
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def mel_spectrogram(y, sr, n_fft=400, hop_length=160, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

if __name__ == "__main__":
    file_path = "/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz/Cargo/12/12_Cargo-Segment_18.wav"
    y, sr = librosa.load(file_path, sr=16000)
    S_dB = mel_spectrogram(y, sr)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=160,
                             x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig("graph/mel_spectrogram.png")

