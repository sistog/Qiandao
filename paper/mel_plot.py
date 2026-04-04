import torch
import torchaudio
import matplotlib.pyplot as plt

# 设置全局字体：英文使用 Times New Roman，中文使用 SimSun
plt.rcParams['font.family'] = 'Times New Roman'  # 默认字体为 Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 处理负号显示问题

# 如果想让中文显示为宋体，则需要配置对应的字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置宋体为中文字体

CARGO_WAV = "/data/zcx/wav_prj/v1/DeepShip/Cargo/27.wav"
PASSENGERSHIP_WAV = "/data/zcx/wav_prj/v1/DeepShip/Passengership/27.wav"
TANKER_WAV = "/data/zcx/wav_prj/v1/DeepShip/Tanker/21.wav"
TUG_WAV = "/data/zcx/wav_prj/v1/DeepShip/Tug/9.wav"

def mel_process(FILE_PATH):
    waveform, sr = torchaudio.load(FILE_PATH)
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
    return mel_spec.T

def plot_mel():
    figure = plt.figure(figsize=(15, 4))

    for file in [CARGO_WAV, PASSENGERSHIP_WAV, TANKER_WAV, TUG_WAV]:
        key = [CARGO_WAV, PASSENGERSHIP_WAV, TANKER_WAV, TUG_WAV].index(file) 
        plt.subplot(1, 4, [CARGO_WAV, PASSENGERSHIP_WAV, TANKER_WAV, TUG_WAV].index(file) + 1)
        mel_spec = mel_process(file)
        title = file.split("/")[-2]
        plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower')
        plt.tick_params(axis='both', labelsize=8) 
        s = chr(key+ord('a'))
        plt.xlabel(f"({s}){title}", fontsize=14)
        # plt.axis('off')  # 隐藏坐标轴
    # plt.tight_layout(pad=0)
    # plt.title(title)
    # plt.xlabel("Time")
    # plt.ylabel("Mel Frequency")
    plt.savefig("assets/Total_mel_spectrogram.png", dpi=300, bbox_inches='tight')  # 保存图像到文件
    plt.show()

if __name__ == "__main__":
    plot_mel()