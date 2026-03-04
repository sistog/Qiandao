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
    def __init__(self, dataset_json_file, label_csv_file, n_fft=8192, transform=None, sr=52734):
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.index_dict = make_index_dict(label_csv_file)
        self.data = data_json['data']
        self.n_fft = n_fft
        self.sr = sr
        self.transform = transform

        if self.transform == "mel":
            self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,  
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

    def __getitem__(self, idx):
        path = self.data[idx]['wav']
        label_str = self.data[idx]['labels']
        label = int(self.index_dict[label_str])

        waveform, sr = torchaudio.load(path)

        if self.transform == "fft":
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
                mel_spec, size=512, mode='linear', align_corners=False
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
            return waveform, torch.tensor(label, dtype=torch.long)



