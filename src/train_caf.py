import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
import time
import os
from model.caf_model import CAF_ViT  # 确保路径正确
from dataset.dataset_caf import CAF_AudioDataset  # 确保路径正确


# --- 2. 适配 CAF 的训练函数 ---
def train_one_epoch_caf(model, dataloader, criterion, optimizer, device, global_step, writer=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(dataloader, desc="Train-CAF")

    for mel, lofar, y in pbar:
        mel, lofar, y = mel.to(device), lofar.to(device), y.to(device)

        optimizer.zero_grad()
        # 双输入传递
        logits = model(mel, lofar)
        loss = criterion(logits, y)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
        # writer.add_scalar('Loss/train', loss.item(), global_step)
    
    return total_loss/total, correct/total, global_step

# --- 3. 适配 CAF 的验证函数 ---
@torch.no_grad()
def validate_caf(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(dataloader, desc="Val-CAF", leave=False)

    for mel, lofar, y in pbar:
        mel, lofar, y = mel.to(device), lofar.to(device), y.to(device)
        logits = model(mel, lofar)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss/total, correct/total

# --- 4. 主程序 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型：dim_a 对应 Mel(128), dim_b 对应 LOFAR(256)
    model = CAF_ViT(dim_a=128, dim_b=256, num_classes=4).to(device)
    
    # 数据准备 (请确保路径正确)
    train_dataset = CAF_AudioDataset(
        "/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_train_data.json",
        "/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_class_map.csv"
    )
        # 建议设置为你的 CPU 核心数（如 8 或 16），或者设置为 4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=16,     # 开启多进程
        pin_memory=True    # 锁页内存，加快从 CPU 到 GPU 的拷贝速度
    )
    
    val_dataset = CAF_AudioDataset(
        "/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_eval_data.json",
        "/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_class_map.csv",
        train = False
    )
    # 建议设置为你的 CPU 核心数（如 8 或 16），或者设置为 4
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=16,     # 开启多进程
        pin_memory=True    # 锁页内存，加快从 CPU 到 GPU 的拷贝速度
    )
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    # writer = SummaryWriter(f"logs/CAF_ViT_{time.time()}")

    global_step = 0
    for epoch in range(20):
        t_loss, t_acc, global_step = train_one_epoch_caf(
            model, train_loader, criterion, optimizer, device, global_step
        )
        v_loss, v_acc = validate_caf(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Train Acc {t_acc:.4f}, Val Acc {v_acc:.4f}")