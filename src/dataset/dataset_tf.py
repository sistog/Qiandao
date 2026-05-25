"""
CAF_AudioDataset — DeepShip 版本
输出三路 T-F 图像，适配 TFCrossAttnModel 的 [B, 1, H, W] 输入格式

三路特征:
    branch 0 — STFT   : [1, 128, 512]
    branch 1 — Mel    : [1, 128, 512]
    branch 2 — CQT    : [1, 128, 512]
"""

import torch
import csv
import json
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset



# ──────────────────────────────────────────────
#  全局归一化参数（可按需调整）
# ──────────────────────────────────────────────
GLOBAL_MIN = torch.tensor(-1.0,           dtype=torch.float32)
GLOBAL_MAX = torch.tensor(0.93310546875,  dtype=torch.float32)


def make_index_dict(label_csv: str) -> dict:
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
    return index_lookup


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Min-max 归一化到 [0, 1]"""
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


class CAF_AudioDataset(Dataset):
    """
    Args:
        dataset_json_file : 训练/验证集 json 路径
        label_csv_file    : 标签 csv 路径
        sr                : 目标采样率（若音频采样率不同则重采样）
        target_freq       : 频率轴高度 H（三路统一）
        target_time       : 时间轴宽度 W（三路统一）
        train             : 是否为训练集（控制数据增强）
        ration            : 小样本比例，0.0 表示使用完整数据集
    """

    def __init__(
        self,
        dataset_json_file: str,
        label_csv_file: str,
        sr: int = 16000,
        target_freq: int = 128,
        target_time: int = 512,
        train: bool = True,
        ration: float = 0.0,
    ):
        # ── 加载数据列表 ──────────────────────────────────────────────
        dataset_file = dataset_json_file
        if ration > 0.0:
            suffix = "train" if train else "val"
            dataset_file = (
                f'/data/zcx/wav_prj/Qiandao/src/datafiles/ration_little/'
                f'deepship_{suffix}_data_{ration:.2f}.json'
            )

        with open(dataset_file, 'r') as fp:
            data_json = json.load(fp)

        self.index_dict   = make_index_dict(label_csv_file)
        self.data         = data_json['data']
        self.sr           = sr
        self.target_freq  = target_freq   # H
        self.target_time  = target_time   # W
        self.train        = train

        # ── STFT 参数 ─────────────────────────────────────────────────
        self.stft_n_fft      = 512
        self.stft_hop_length = 128
        self.stft_win_length = 512

        # ── Mel 参数 ──────────────────────────────────────────────────
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=target_freq,          # 直接输出 target_freq 个 Mel bin
        )

        # ── CQT 参数（用 torchaudio VQT 近似 CQT） ────────────────────
        # torchaudio 0.13+ 提供 torchaudio.transforms.CQT 但不稳定；
        # 这里用等效的 STFT 近似：对数频率轴重采样模拟 CQT 的对数分辨率
        self.cqt_n_fft      = 2048
        self.cqt_hop_length = 512
        self.cqt_win_length = 2048
        self.n_bins         = target_freq   # CQT 输出频率维度

    # ──────────────────────────────────────────────────────────────────
    #  特征提取方法
    # ──────────────────────────────────────────────────────────────────

    def _get_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        STFT → log 幅度谱
        输出: [1, target_freq, target_time]
        """
        waveform_1d = waveform.squeeze(0)   # [samples]
        spec = torch.stft(
            waveform_1d,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            window=torch.hann_window(self.stft_win_length, device=waveform.device),
            return_complex=True,
        )                                   # [F, T]  F = n_fft//2 + 1
        spec = torch.log(torch.abs(spec) + 1e-6)   # log 压缩
        spec = spec.unsqueeze(0).unsqueeze(0)       # [1, 1, F, T]
        spec = F.interpolate(
            spec,
            size=(self.target_freq, self.target_time),
            mode='bilinear',
            align_corners=False,
        )                                           # [1, 1, H, W]
        return spec.squeeze(0)                      # [1, H, W]

    def _get_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        # 1. 提取 Fbank 特征
        # waveform shape 预期为 [1, T]
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=16000,
            use_log_fbank=True,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_length=25,
            frame_shift=10
        ) # 返回 [n_frames, 128]

        # 2. 调整维度以适配 interpolate (Batch, Channel, H, W)
        # 转换为 [1, 1, 128, n_frames]
        fbank = fbank.transpose(0, 1).unsqueeze(0).unsqueeze(0)

        # 3. 插值缩放
        fbank_resized = F.interpolate(
            fbank, 
            size=(self.target_freq, self.target_time), 
            mode='bilinear',
            align_corners=False
        )

        # 4. 归一化 (可选，建议添加，例如针对当前 batch 的归一化)
        # fbank_resized = (fbank_resized - fbank_resized.mean()) / (fbank_resized.std() + 1e-6)

        # 5. 返回 [1, H, W]
        return fbank_resized.squeeze(1) # 从 [1, 1, H, W] 变为 [1, H, W]

    def _get_cqt(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        CQT 近似：在线性频率 STFT 上做对数频率轴重采样
        （等效于将线性频率轴映射到对数刻度，模拟 CQT 的等 Q 特性）
        输出: [1, target_freq, target_time]
        """
        waveform_1d = waveform.squeeze(0)
        spec = torch.stft(
            waveform_1d,
            n_fft=self.cqt_n_fft,
            hop_length=self.cqt_hop_length,
            win_length=self.cqt_win_length,
            window=torch.hann_window(self.cqt_win_length, device=waveform.device),
            return_complex=True,
        )                                           # [F_linear, T]
        spec = torch.abs(spec)

        F_linear, T = spec.shape

        # 对数频率轴重采样：从线性频率轴采样 n_bins 个对数均匀间隔的频点
        # freq_idx[i] ∈ [1, F_linear)，对数均匀分布
        log_freq_idx = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(1.0)),
                torch.log(torch.tensor(float(F_linear - 1))),
                self.n_bins,
            )
        ).long().clamp(0, F_linear - 1)             # [n_bins]

        cqt = spec[log_freq_idx, :]                 # [n_bins, T]
        cqt = torch.log(cqt + 1e-6)

        # 时间轴对齐
        cqt = cqt.unsqueeze(0).unsqueeze(0)         # [1, 1, n_bins, T]
        cqt = F.interpolate(
            cqt,
            size=(self.target_freq, self.target_time),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)                                 # [1, H, W]
        return cqt

    # ──────────────────────────────────────────────────────────────────
    #  数据增强
    # ──────────────────────────────────────────────────────────────────

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # 随机加噪
        if torch.rand(1) < 0.2:
            waveform = waveform + torch.randn_like(waveform) * 0.02
        # 随机音量缩放
        if torch.rand(1) < 0.5:
            scale = torch.rand(1) * 0.4 + 0.8      # [0.8, 1.2]
            waveform = waveform * scale
        return waveform

    # ──────────────────────────────────────────────────────────────────
    #  Dataset 接口
    # ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns:
            stft_img  : Tensor [1, target_freq, target_time]
            mel_img   : Tensor [1, target_freq, target_time]
            cqt_img   : Tensor [1, target_freq, target_time]
            label     : LongTensor []
        """
        # ── 加载音频 ──────────────────────────────────────────────────
        path      = self.data[idx]['wav']
        label_str = self.data[idx]['labels']
        label     = int(self.index_dict[label_str])

        waveform, file_sr = torchaudio.load(path)   # [C, samples]

        # 重采样（如有必要）
        if file_sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, file_sr, self.sr)

        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 数据增强
        if self.train:
            waveform = self._augment(waveform)

        # ── 提取三路特征 ──────────────────────────────────────────────
        stft_img = normalize(self._get_stft(waveform))   # [1, H, W]
        mel_img  = normalize(self._get_mel(waveform))    # [1, H, W]
        cqt_img  = normalize(self._get_cqt(waveform))   # [1, H, W]

        return stft_img, mel_img, cqt_img, torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────
#  DataLoader 工厂函数（便于直接调用）
# ──────────────────────────────────────────────────────────────────────

def build_dataloaders(
    train_json: str,
    val_json: str,
    label_csv: str,
    sr: int = 52734,
    target_freq: int = 128,
    target_time: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    ration: float = 0.0,
):
    train_ds = CAF_AudioDataset(
        train_json, label_csv,
        sr=sr, target_freq=target_freq, target_time=target_time,
        train=True, ration=ration,
    )
    val_ds = CAF_AudioDataset(
        val_json, label_csv,
        sr=sr, target_freq=target_freq, target_time=target_time,
        train=False, ration=ration,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────
#  Smoke test
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # 用随机波形验证输出 shape（不依赖真实文件）
    class _DummyDataset(CAF_AudioDataset):
        def __init__(self):
            # 跳过文件 IO，直接初始化参数
            self.sr           = 52734
            self.target_freq  = 128
            self.target_time  = 512
            self.train        = True
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_fft=2048, win_length=2048,
                hop_length=512, n_mels=self.target_freq,
            )
            self.stft_n_fft = 512;  self.stft_hop_length = 128;  self.stft_win_length = 512
            self.cqt_n_fft  = 2048; self.cqt_hop_length  = 512;  self.cqt_win_length  = 2048
            self.n_bins = self.target_freq
            self.data = [None] * 4
            self.index_dict = {}

        def __getitem__(self, idx):
            waveform = torch.randn(1, self.sr * 5)    # 5 秒随机信号
            stft_img = normalize(self._get_stft(waveform))
            mel_img  = normalize(self._get_mel(waveform))
            cqt_img  = normalize(self._get_cqt(waveform))
            label    = torch.tensor(0, dtype=torch.long)
            return stft_img, mel_img, cqt_img, label

    ds = _DummyDataset()
    stft, mel, cqt, lbl = ds[0]
    print(f"STFT shape : {stft.shape}")   # [1, 128, 512]
    print(f"Mel  shape : {mel.shape}")    # [1, 128, 512]
    print(f"CQT  shape : {cqt.shape}")   # [1, 128, 512]
    print(f"Label      : {lbl}")
    print("Dataset smoke test OK ✓")