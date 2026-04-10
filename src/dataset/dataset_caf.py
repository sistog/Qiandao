import torch
import csv
import json
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

# 保持你之前的全局归一化参数
GLOBAL_MIN = torch.tensor(-1.0, dtype=torch.float32)
GLOBAL_MAX = torch.tensor(0.93310546875, dtype=torch.float32)

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
    return index_lookup

class CAF_AudioDataset(Dataset):
    def __init__(self, dataset_json_file, label_csv_file, sr=52734, target_length=512, ration=0.0, train=True):
        dataset_file = dataset_json_file
        if ration > 0.0:
            suffix = "train" if train else "val"
            dataset_file = f'/data/zcx/wav_prj/Qiandao/src/datafiles/ration_little/deepship_{suffix}_data_{ration:.2f}.json'
        
        with open(dataset_file, 'r') as fp:
            data_json = json.load(fp)
            
        self.index_dict = make_index_dict(label_csv_file)
        self.data = data_json['data']
        self.sr = sr
        self.target_length = target_length  # 对应 Transformer 的序列长度 (Time frames)
        self.train = train

        # 定义 Mel 转换
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=128
        )

    def get_lofar(self, waveform):
        """
        计算 LOFAR 谱图：使用大窗口 STFT 获取高分辨率线谱
        """
        n_fft_lofar = 4096 
        hop_lofar = 512
        
        spec = torch.stft(
            waveform, 
            n_fft=n_fft_lofar, 
            hop_length=hop_lofar, 
            win_length=n_fft_lofar,
            window=torch.hann_window(n_fft_lofar).to(waveform.device),
            return_complex=True
        )
        # 取模并进行对数压缩
        lofar = torch.log10(torch.abs(spec) + 1e-6)
        # 取低频部分 (0-256个频点)，对应 CAF 模型要求的 256 维输入
        lofar = lofar[:, :256, :] 
        return lofar

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]['wav']
        label_str = self.data[idx]['labels']
        label = int(self.index_dict[label_str])

        waveform, _ = torchaudio.load(path)
        
        # 数据增强（仅在训练时应用）
        if self.train:
            # # 随机音量变化
            # if torch.rand(1) < 0.5:
            #     volume_factor = torch.rand(1) * 0.4 + 0.8  # 0.8-1.2
            #     waveform = waveform * volume_factor
            
            # 随机添加噪声
            if torch.rand(1) < 0.2:
                noise = torch.randn_like(waveform) * 0.02
                waveform = waveform + noise

        # ---- 支路 A: Mel Spectrogram [128 维] ----
        mel_spec = self.mel_transform(waveform) # 形状: [1, 128, T]
        mel_spec = torch.log(mel_spec + 1e-6)
        
        # 使用 bilinear 模式缩放，输入需为 4D [N, C, H, W]
        # 我们将 H 固定为 128 (Mel bins)，将 W 缩放为 target_length (512)
        mel_spec = F.interpolate(
            mel_spec.unsqueeze(0), 
            size=(128, self.target_length), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0) # 结果: [1, 128, 512]

        # ---- 支路 B: LOFAR Spectrogram [256 维] ----
        lofar_spec = self.get_lofar(waveform) # 形状: [1, 256, T]
        
        lofar_spec = F.interpolate(
            lofar_spec.unsqueeze(0), 
            size=(256, self.target_length), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0) # 结果: [1, 256, 512]

        # ---- 格式转换：适配 Transformer 的 [Sequence, Feature] 输入 ----
        # 最终输出需要去掉 Batch/Channel 维，并转置为 [Time, Dim]
        mel_out = mel_spec.squeeze(0).transpose(0, 1)    # [512, 128]
        lofar_out = lofar_spec.squeeze(0).transpose(0, 1) # [512, 256]

        return mel_out, lofar_out, torch.tensor(label, dtype=torch.long)