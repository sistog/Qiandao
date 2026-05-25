# -*- coding: utf-8 -*-
"""
音频傅里叶变换分析工具
- FFT 频谱图
- STFT 语谱图（时频图）
- 功率谱密度
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf


def read_audio(file_path):
    """读取音频文件"""
    data, sr = sf.read(file_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # 多声道转单声道
    return data, sr


def plot_fft(data, sr, save_path=None):
    """绘制 FFT 频谱"""
    n = len(data)
    yf = np.fft.rfft(data)
    xf = np.fft.rfftfreq(n, 1 / sr)

    plt.figure(figsize=(12, 6))
    plt.plot(xf, np.abs(yf))
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"FFT spectrum saved to {save_path}")
    plt.show()
    plt.close()


def plot_spectrogram(data, sr, save_path=None):
    """绘制 STFT 语谱图"""
    plt.figure(figsize=(12, 6))
    D = librosa.stft(data)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(img, format="%+2.0f dB")
    plt.title("Spectrogram (STFT)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Spectrogram saved to {save_path}")
    plt.show()
    plt.close()


def plot_mel_spectrogram(data, sr, save_path=None):
    """绘制 Mel 语谱图"""
    plt.figure(figsize=(12, 6))
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(img, format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Mel spectrogram saved to {save_path}")
    plt.show()
    plt.close()


def plot_power_spectrum(data, sr, save_path=None):
    """绘制功率谱密度"""
    n = len(data)
    yf = np.fft.rfft(data)
    psd = (np.abs(yf) ** 2) / n
    xf = np.fft.rfftfreq(n, 1 / sr)

    plt.figure(figsize=(12, 6))
    plt.semilogy(xf, psd)
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power / Frequency (dB)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Power spectrum saved to {save_path}")
    plt.show()
    plt.close()


def plot_time_domain(data, sr, save_path=None):
    """绘制时域波形"""
    t = np.linspace(0, len(data) / sr, len(data))

    plt.figure(figsize=(12, 4))
    plt.plot(t, data)
    plt.title("Time Domain Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Time domain plot saved to {save_path}")
    plt.show()
    plt.close()


def analyze_audio(file_path, out_dir=None):
    """全流程分析"""
    if out_dir is None:
        out_dir = os.path.dirname(file_path) or "."

    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading audio: {file_path}")
    data, sr = read_audio(file_path)
    duration = len(data) / sr
    print(f"Sample rate: {sr} Hz, Duration: {duration:.2f}s, Samples: {len(data)}")

    base = os.path.splitext(os.path.basename(file_path))[0]

    plot_time_domain(data, sr, save_path=os.path.join(out_dir, f"{base}_time.png"))
    plot_fft(data, sr, save_path=os.path.join(out_dir, f"{base}_fft.png"))
    plot_spectrogram(data, sr, save_path=os.path.join(out_dir, f"{base}_spectrogram.png"))
    plot_mel_spectrogram(data, sr, save_path=os.path.join(out_dir, f"{base}_mel.png"))
    plot_power_spectrum(data, sr, save_path=os.path.join(out_dir, f"{base}_power_spectrum.png"))

    # 打印频域统计信息
    n = len(data)
    yf = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    magnitudes = np.abs(yf)

    dominant_freq = freqs[np.argmax(magnitudes)]
    print(f"\nDominant frequency: {dominant_freq:.2f} Hz")
    print(f"Total energy: {np.sum(magnitudes ** 2):.2f}")
    print("\nAll plots generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Fourier Transform Analysis")
    parser.add_argument("--input", type=str,
                        default=os.path.join(os.path.dirname(__file__), "raw_audio.wav"),
                        help="Input audio file path")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for plots (default: same as input)")
    args = parser.parse_args()

    analyze_audio(args.input, args.out_dir)
