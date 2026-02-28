import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

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


plt.figure(figsize=(10, 4))
plt.imshow(mel_spec_db.numpy(), origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.xlabel("Time Frames")
plt.ylabel("Mel Bins")
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.savefig("/data/zcx/wav_prj/Qiandao/assets/output_images/mel_spectrogram.png", dpi=300)
plt.show()
