import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

import librosa
import librosa.display
import numpy as np

wav_file = "/data/zcx/wav_prj/ast/egs/UUV/dataset/other/20220624110853_S_No7_F_W_0_label__0__9.wav"
waveform, sr = torchaudio.load(wav_file)  # waveform: [1, num_samples]
print(sr)

# Mel Spectrogram
mel_spec_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=8192,
    win_length=8192,
    hop_length=512,
    n_mels=128
)
mel_spec = mel_spec_transform(waveform)  # [1, n_mels, n_frames]

# 将最后一维时间维 resize 为 2048
mel_spec_resized = F.interpolate(mel_spec, size=2048, mode='linear', align_corners=False)
# shape: [1, 128, 2048]
print(mel_spec_resized.shape)



# 假设 mel_spec_resized: [1, 128, 2048]
mel_spec = mel_spec_resized[0]  # 去掉 batch 维 → [128, 2048]

# 可选：转换为 dB
mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)

os.makedirs("/data/zcx/wav_prj/Qiandao/assets/output_images", exist_ok=True)

waveform = waveform.mean(dim=0)   # 单通道
waveform = waveform.numpy()       # 转 numpy
y=waveform

C = librosa.cqt(
    y,
    sr=sr,
    hop_length=512,
    fmin=20,
    n_bins=96,
    bins_per_octave=12
)

C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
# C_db 是 numpy 数组 [96, 309]

C_tensor = torch.tensor(C_db).float()     # [96, 309]

C_tensor = C_tensor.unsqueeze(0).unsqueeze(0)
# 变成 [1, 1, 96, 309]
# 格式: [B, C, H, W]

C_resized = F.interpolate(
    C_tensor,
    size=(128, 512),
    mode='bilinear',
    align_corners=False
)

C_resized = C_resized.squeeze(0).squeeze(0)
print(C_resized.shape)  # (128, 512)

plt.figure(figsize=(10, 6))
librosa.display.specshow(
    C_resized.numpy(),
    sr=sr,
    hop_length=512,
    x_axis='time',
    y_axis='cqt_hz'
)
plt.colorbar(format='%+2.0f dB')
plt.title("CQT Spectrogram")
plt.tight_layout()
plt.savefig("/data/zcx/wav_prj/Qiandao/assets/output_images/cqt_spectrogram.png", dpi=300)
plt.show()


# plt.figure(figsize=(10, 4))
# plt.imshow(mel_spec_db.numpy(), origin='lower', aspect='auto', cmap='viridis')
# plt.colorbar(format='%+2.0f dB')
# plt.xlabel("Time Frames")
# plt.ylabel("Mel Bins")
# plt.title("Mel Spectrogram")
# plt.tight_layout()
# plt.savefig("/data/zcx/wav_prj/Qiandao/assets/output_images/mel_spectrogram.png", dpi=300)
# plt.show()
