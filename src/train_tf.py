"""
训练脚本 — TFCrossAttnModel (STFT / Mel / CQT 三路融合)
对应数据集: DeepShip
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model.TFCross_model import TFCrossAttnModel, MultiHeadLoss   # 你的模型文件
from dataset.dataset_tf import CAF_AudioDataset              # 你的数据集文件


# ─────────────────────────────────────────────
#  配置
# ─────────────────────────────────────────────

CONFIG = dict(
    # 路径
    train_json  = "/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_train_data.json",
    val_json    = "/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_eval_data.json",
    label_csv   = "/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_class_map.csv",
    save_dir    = "checkpoints",

    # 数据
    sr          = 16000,
    target_freq = 128,
    target_time = 512,
    batch_size  = 16,
    num_workers = 4,

    # 模型
    num_branches = 3,
    num_classes  = 4,
    patch_size   = 16,
    embed_dim    = 256,
    num_heads    = 4,
    depth        = 4,
    mlp_ratio    = 2.0,
    dropout      = 0.1,

    # 训练
    epochs       = 50,
    lr           = 1e-3,
    weight_decay = 1e-5,
    lambda_kl    = 0.01,       # MultiHeadLoss KL 正则项权重
    grad_clip    = 5.0,        # 梯度裁剪上限，None 表示不裁剪

    # 学习率调度
    lr_scheduler = "cosine",   # "cosine" | "step" | None
    warmup_epochs = 1,         # 线性 warmup 轮数

    # 其他
    ration      = 0.0,         # 小样本比例，0.0 = 完整数据集
    seed        = 42,
)


# ─────────────────────────────────────────────
#  工具
# ─────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, cfg, steps_per_epoch: int):
    """返回 (scheduler, 调用时机) 二元组，时机为 'epoch' 或 'step'"""
    if cfg['lr_scheduler'] is None:
        return None, None

    warmup_steps = cfg['warmup_epochs'] * steps_per_epoch

    if cfg['lr_scheduler'] == 'cosine':
        # 先线性 warmup，再 cosine 衰减
        total_steps = cfg['epochs'] * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler, 'step'

    if cfg['lr_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
        return scheduler, 'epoch'

    return None, None


# ─────────────────────────────────────────────
#  训练 / 验证
# ─────────────────────────────────────────────

def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler,
    sched_timing, device, global_step, writer, cfg
):
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(dataloader, desc="Train", leave=True)
    for stft, mel, cqt, y in pbar:
        stft, mel, cqt, y = (
            stft.to(device), mel.to(device), cqt.to(device), y.to(device)
        )

        optimizer.zero_grad()

        output = model([stft, mel, cqt])
        loss   = criterion(output, y)

        loss.backward()

        if cfg['grad_clip']:
            nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])

        optimizer.step()

        if scheduler and sched_timing == 'step':
            scheduler.step()

        # 用 final_probs 计算准确率
        preds = output['final_probs'].argmax(dim=1)
        batch_n = y.size(0)
        total_loss += loss.item() * batch_n
        correct    += (preds == y).sum().item()
        total      += batch_n
        global_step += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct/total:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        if writer:
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

    return total_loss / total, correct / total, global_step


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    pbar = tqdm(dataloader, desc="Val  ", leave=False)
    for stft, mel, cqt, y in pbar:
        stft, mel, cqt, y = (
            mel.to(device), mel.to(device), mel.to(device), y.to(device)
        )

        output = model([stft, mel, cqt])
        loss   = criterion(output, y)

        preds = output['final_probs'].argmax(dim=1)
        batch_n = y.size(0)
        total_loss += loss.item() * batch_n
        correct    += (preds == y).sum().item()
        total      += batch_n

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = CONFIG
    set_seed(cfg['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg['save_dir'], exist_ok=True)

    # ── 数据集 ────────────────────────────────────────────────────────
    train_dataset = CAF_AudioDataset(
        dataset_json_file = cfg['train_json'],
        label_csv_file    = cfg['label_csv'],
        sr                = cfg['sr'],
        target_freq       = cfg['target_freq'],
        target_time       = cfg['target_time'],
        train             = True,
        ration            = cfg['ration'],
    )
    val_dataset = CAF_AudioDataset(
        dataset_json_file = cfg['val_json'],
        label_csv_file    = cfg['label_csv'],
        sr                = cfg['sr'],
        target_freq       = cfg['target_freq'],
        target_time       = cfg['target_time'],
        train             = False,
        ration            = cfg['ration'],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = True,
        num_workers = cfg['num_workers'],
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = cfg['num_workers'],
        pin_memory  = True,
    )
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # ── 模型 ──────────────────────────────────────────────────────────
    model = TFCrossAttnModel(
        num_branches = cfg['num_branches'],
        num_classes  = cfg['num_classes'],
        img_size     = (cfg['target_freq'], cfg['target_time']),
        patch_size   = cfg['patch_size'],
        in_channels  = 1,
        embed_dim    = cfg['embed_dim'],
        num_heads    = cfg['num_heads'],
        depth        = cfg['depth'],
        mlp_ratio    = cfg['mlp_ratio'],
        dropout      = cfg['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── 损失 / 优化器 / 调度器 ────────────────────────────────────────
    criterion = MultiHeadLoss(num_classes=cfg['num_classes'], lambda_kl=cfg['lambda_kl'])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler, sched_timing = get_lr_scheduler(optimizer, cfg, len(train_loader))

    # ── TensorBoard ───────────────────────────────────────────────────
    writer = SummaryWriter(f"logs/TFCrossAttn_{time.strftime('%Y%m%d_%H%M%S')}")

    # ── 训练循环 ──────────────────────────────────────────────────────
    best_val_acc  = 0.0
    global_step   = 0

    for epoch in range(1, cfg['epochs'] + 1):
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{cfg['epochs']}")

        t_loss, t_acc, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, sched_timing, device, global_step, writer, cfg
        )

        v_loss, v_acc = validate(model, val_loader, criterion, device)

        if scheduler and sched_timing == 'epoch':
            scheduler.step()

        # 记录到 TensorBoard
        writer.add_scalars('Loss', {'train': t_loss, 'val': v_loss}, epoch)
        writer.add_scalars('Acc',  {'train': t_acc,  'val': v_acc},  epoch)

        print(
            f"  Train — loss: {t_loss:.4f}  acc: {t_acc:.4f}\n"
            f"  Val   — loss: {v_loss:.4f}  acc: {v_acc:.4f}"
        )

        # ── 保存最优模型 ──────────────────────────────────────────────
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            ckpt_path = os.path.join(cfg['save_dir'], "best_model.pth")
            torch.save(
                {
                    'epoch'     : epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'val_acc'   : v_acc,
                    'val_loss'  : v_loss,
                    'config'    : cfg,
                },
                ckpt_path,
            )
            print(f"  ✓ Best model saved → {ckpt_path}  (val_acc={v_acc:.4f})")

        # ── 定期保存 checkpoint ───────────────────────────────────────
        if epoch % 10 == 0:
            ckpt_path = os.path.join(cfg['save_dir'], f"epoch_{epoch:03d}.pth")
            torch.save(
                {
                    'epoch'     : epoch,
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'val_acc'   : v_acc,
                    'config'    : cfg,
                },
                ckpt_path,
            )

    writer.close()
    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")