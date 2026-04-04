import torchaudio
import torch
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

EXAMPLE_WAV = "/data/zcx/wav_prj/v1/DeepShip/Cargo/27.wav"
waveform, sr = torchaudio.load(EXAMPLE_WAV)
print(f"Sample rate: {sr}, Waveform shape: {waveform.shape}")

# Path("assets").mkdir(exist_ok=True)  # 创建 assets 目录用于保存图像


# 绘制波形图
plt.figure(figsize=(12, 6))
plt.plot(waveform.t().numpy())
plt.axis('off')  # 隐藏坐标轴
plt.tight_layout(pad=0)
# plt.title("Waveform")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
plt.savefig("assets/waveform.png")  # 保存图像到文件
plt.show()

# 绘制梅尔频谱图
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=8192,
    win_length=8192,    
    hop_length=512,
    n_mels=128
)
mel_spec = mel_spec_transform(waveform)
mel_spec = mel_spec.squeeze(0)  # 去掉批次维度
mel_spec = mel_spec.log2()
mel_spec = mel_spec.T

print(f"Mel Spectrogram shape: {mel_spec.shape}")
plt.figure(figsize=(12, 6))
plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower')
plt.axis('off')  # 隐藏坐标轴
plt.tight_layout(pad=0)
# plt.title("Mel Spectrogram")
# plt.xlabel("Time")
# plt.ylabel("Mel Frequency")
plt.savefig("assets/mel_spectrogram.png")  # 保存图像到文件
plt.show()

