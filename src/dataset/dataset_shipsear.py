import torch
import csv
import json
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F



def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

class AudioDataset(Dataset):
    def __init__(self, dataset_json_file, label_csv_file, n_fft=8192, transform=None, sr=52734, ration=0.0, train=True):
        dataset_file = dataset_json_file
        if ration > 0.0:
            if train:
                dataset_file = f'/data/zcx/wav_prj/Qiandao/src/datafiles/ration_little/deepship_train_data_{ration:.2f}.json'
            else:
                dataset_file = f'/data/zcx/wav_prj/Qiandao/src/datafiles/ration_little/deepship_val_data_{ration:.2f}.json'
        with open(dataset_file, 'r') as fp:
            data_json = json.load(fp)
        self.index_dict = make_index_dict(label_csv_file)
        self.data = data_json['data']
        self.n_fft = n_fft
        self.sr = sr
        self.transform = transform
        self.train = train
        self.target_length = 512
        # if self.transform == "mel":
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,  
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=512,
            n_mels=128
        )
            
        # if self.transform == 'fbank':
        #     self.fbank_transform = torchaudio.compliance.kaldi.fbank(
        #         waveform,
        #         sample_frequency=self.sr,
        #         use_log_fbank=True,
        #         use_energy=False,
        #         window_type='hanning',
        #         num_mel_bins=128,
        #         dither=0.0,
        #         frame_shift=10
        #     )
    def __len__(self):
        return len(self.data)
    
    def get_lofar(self, waveform):
        """
        计算 LOFAR 谱图 (通常使用大点数 FFT 提取线谱)
        """
        # 水下 LOFAR 常用的参数：大窗口以获得高频率分辨率
        n_fft_lofar = 4096 
        hop_lofar = 512
        
        # 使用 STFT 计算功率谱
        spec = torch.stft(
            waveform, 
            n_fft=n_fft_lofar, 
            hop_length=hop_lofar, 
            win_length=n_fft_lofar,
            window=torch.hann_window(n_fft_lofar).to(waveform.device),
            return_complex=True
        )
        lofar = torch.abs(spec)
        # 对数压缩
        lofar = torch.log10(lofar + 1e-6)
        # 取低频部分 (根据 DeepShip 采样率，通常关注低频段)
        # 比如只取前 256 个频点
        lofar = lofar[:, :256, :] 
        return lofar

    def __getitem__(self, idx):
        path = self.data[idx]['wav']
        label_str = self.data[idx]['labels']
        label = int(self.index_dict[label_str])

        waveform, sr = torchaudio.load(path)
        # waveform = (waveform - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)

        if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
        # ensure mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if self.transform == 'caf':
            # ---- 特征 1: Mel 谱图 ----
            mel_spec = self.mel_spec_transform(waveform) # [C, 128, T]
            mel_spec = torch.log(mel_spec + 1e-6)
            # 统一长度到 target_length
            mel_spec = F.interpolate(mel_spec, size=self.target_length, mode='linear', align_corners=False)
            mel_spec = mel_spec.squeeze(0).transpose(0, 1) # [T, 128]

            # ---- 特征 2: LOFAR 谱图 ----
            lofar_spec = self.get_lofar(waveform) # [C, F, T]
            lofar_spec = F.interpolate(lofar_spec, size=self.target_length, mode='linear', align_corners=False)
            lofar_spec = lofar_spec.squeeze(0).transpose(0, 1) # [T, F_lofar]

            # 返回双特征及标签
            return mel_spec, lofar_spec, torch.tensor(label, dtype=torch.long)

        elif self.transform == "fft":
            # 补零或截断
            if waveform.size(1) < self.n_fft:
                pad = self.n_fft - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            else:
                waveform = waveform[:, :self.n_fft]

            fft_feat = torch.fft.fft(waveform, n=self.n_fft)
            fft_feat = torch.abs(fft_feat).float()
            return fft_feat, torch.tensor(label, dtype=torch.long)

        elif self.transform == "mel":
            mel_spec = self.mel_spec_transform(waveform)  # [1, 128, n_frames] [C, Mel_Bins, T]
            mel_spec = torch.log(mel_spec + 1e-6)
            mel_spec_resized = F.interpolate(
                mel_spec, size=256, mode='linear', align_corners=False
            )
            return mel_spec_resized, torch.tensor(label, dtype=torch.long)
        elif self.transform == "fbank":
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                sample_frequency=self.sr,
                use_log_fbank=True,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_length=25,  # 25ms
                frame_shift=10  # [n_frames, n_mel_bins]
            )
            fbank = fbank.transpose(0, 1).unsqueeze(0)  # [1, n_mel_bins, n_frames]
            fbank_resized = F.interpolate(
                fbank, size=512, mode='linear', align_corners=False
            )
            # 最终输出格式[C, F, T]
            return fbank_resized, torch.tensor(label, dtype=torch.long)
        elif self.transform == 'ast':
            target_length = 512
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=self.sr,
                window_type='hanning',
                use_energy=False,
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            )

            n_frames = fbank.shape[0]

            if n_frames < target_length:
                fbank = torch.nn.functional.pad(
                    fbank,
                    (0, 0, 0, target_length - n_frames)
                )
            else:
                fbank = fbank[:target_length, :]
            
            norm_mean = 0
            norm_std = 1
            freqm = 24
            timem = 96

             # SpecAug, not do for eval set
            freqm = torchaudio.transforms.FrequencyMasking(freqm)
            timem = torchaudio.transforms.TimeMasking(timem)
            fbank = torch.transpose(fbank, 0, 1)
            # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
            fbank = fbank.unsqueeze(0)
            if freqm != 0:
                fbank = freqm(fbank)
            if timem != 0:
                fbank = timem(fbank)
            # squeeze it back, it is just a trick to satisfy new torchaudio version
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)
            
            fbank = (fbank - norm_mean) / (norm_std + 1e-5)

             # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
            return fbank, torch.tensor(label, dtype=torch.long)
        elif self.transform == 'raw':        
            
            
            # 数据增强（仅在训练时应用）
            if self.train:
                # 随机音量变化
                if torch.rand(1) < 0.5:
                    volume_factor = torch.rand(1) * 0.4 + 0.8  # 0.8-1.2
                    waveform = waveform * volume_factor
                
                # 随机添加噪声
                if torch.rand(1) < 0.2:
                    noise = torch.randn_like(waveform) * 0.02
                    waveform = waveform + noise
            
            return waveform, torch.tensor(label, dtype=torch.long)
          



