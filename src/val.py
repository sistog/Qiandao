import torch
import time
import os
import torch.nn as nn 
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.CNN1D_NET import AudioCNN1D
from model.CNN2D_NET import AudioCNN2D
from model.LSTM_NET import AudioLSTM
from model.ResNet import ResNetAudio
from model.ViT_model import AcousticViT
from dataset.qiandao_dataset import AudioDataset
from model.ast_models import ASTModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from collections import defaultdict
import torchaudio
import torch.nn.functional as F



CLASS_NAMES = ['Cargo', 'Passengership', 'Tanker', 'Tug']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASS_NAMES)}
transform = 'ast'
n_fft = 8192

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTModel(
        label_dim=4,
        fstride=10,
        tstride=10,
        input_fdim=128,
        input_tdim=512,
        imagenet_pretrain=True,
        audioset_pretrain=True,
        model_size='base384'
        ).to(device)
    ckpt_path = "/data/zcx/wav_prj/Qiandao/src/exp/Deepship/ckpt/ast_best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print("模型加载成功")
    return model
    

# ===============================
# 2️⃣ 音频预处理
# ===============================
def preprocess_audio(audio_path, target_length=512):
    waveform, sr = torchaudio.load(audio_path)

    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,  
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=512,
                n_mels=128
            )
            

    

    if transform == "fft":
        # 补零或截断
        if waveform.size(1) < n_fft:
            pad = n_fft - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :n_fft]

        fft_feat = torch.fft.fft(waveform, n=n_fft)
        fft_feat = torch.abs(fft_feat).float()
        return fft_feat

    elif transform == "mel":
        mel_spec = mel_spec_transform(waveform)  # [1, 128, n_frames] [C, Mel_Bins, T]
        mel_spec = torch.log(mel_spec + 1e-6)
        mel_spec_resized = F.interpolate(
            mel_spec, size=512, mode='linear', align_corners=False
        )
        return mel_spec_resized
    elif transform == "fbank":
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=sr,
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
        return fbank_resized
    elif transform == "ast":
        target_length = 512
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=16000,
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
        return fbank.unsqueeze(0)





def predict_single(model, audio_path, device):
    with torch.no_grad():
        input_tensor = preprocess_audio(audio_path).to(device)
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

    pred_prob, pred_idx = probs.max(dim=1)
    return pred_idx.item()


def predict_on_folder(model, folder_path, device):
    results = []
    print(f"🚀 开始评估，共 {len(CLASS_NAMES)} 个类别")
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    total_correct = 0
    total_samples = 0

    for class_name in CLASS_NAMES:  # 只评估这三个类别
        class_dir = Path(folder_path) / class_name
        if not class_dir.is_dir():
            print(f"⚠️ 类别文件夹不存在：{class_dir}")
            continue

        true_idx = CLASS_TO_IDX[class_name]

        print(f"\n🔍 评估类别：{class_name}")
        audio_files = list(class_dir.rglob("*.wav"))
        for audio_file in tqdm(audio_files, desc=class_name):
            pred_idx = predict_single(model, str(audio_file), device)
            pred_class = IDX_TO_CLASS[pred_idx]
            if pred_idx == true_idx:
                class_correct[class_name] += 1
                total_correct += 1
            class_total[class_name] += 1
            total_samples += 1
            print(f"{audio_file.name}: 真实类别={class_name}, 预测类别={pred_class}")
        print("\n📊 各类别准确率：")

    for class_name in CLASS_NAMES:
        if class_total[class_name] > 0:
            acc = class_correct[class_name] / class_total[class_name]
            print(f"{class_name}: {class_correct[class_name]}/{class_total[class_name]} = {acc:.2%}")
        else:
            print(f"{class_name}: 无样本")

    # ==============================
    # 计算总体准确率
    # ==============================

    if total_samples > 0:
        overall_acc = total_correct / total_samples
        print(f"\n🎯 总体准确率: {total_correct}/{total_samples} = {overall_acc:.2%}")
    else:
        print("⚠️ 没有有效样本")


if __name__ == "__main__":
    fold_path = "/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz"
    model = load_model()
    predict_on_folder(model, folder_path=fold_path, device='cuda')