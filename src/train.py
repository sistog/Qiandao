import torch
import time
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn 
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.CNN1D_NET import AudioCNN1D
from model.CNN2D_NET import AudioCNN2D
from model.LSTM_NET import AudioLSTM
from model.ResNet import ResNetAudio
from model.ViT_model import AcousticViT
from model.ast_models import ASTModel
from model.Beats.Beats_Transfer import BEATsTransferLearningModel
from model.WavLm.WavLM_Classfier import WavLMClassifier
from dataset.qiandao_dataset import AudioDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from torch.utils.tensorboard import SummaryWriter



def train_one_epoch(model, dataloader, criterion, optimizer, device, global_step, writer):
    
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train")

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # print("x.shape:", x.shape)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        global_step += 1

        # tqdm 实时显示
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}"
        )
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('Accuracy/train', correct / total, global_step)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, global_step


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # 类别正确预测数和样本数
    correct_per_class = [0] * 4  # 假设有4个类别，具体数量根据你的模型调整
    total_per_class = [0] * 4   # 假设有4个类别，具体数量根据你的模型调整

    pbar = tqdm(dataloader, desc="Val", leave=False)

    with torch.no_grad():   # ⭐ 非常重要
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            # 计算每个类别的正确预测数
            for i in range(len(y)):
                correct_per_class[y[i].item()] += (preds[i] == y[i]).item()
                total_per_class[y[i].item()] += 1

            # 实时显示
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}"
            )

    avg_loss = total_loss / total
    accuracy = correct / total

    # 计算AA
    aa = sum([correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0 for i in range(len(correct_per_class))]) / len(correct_per_class)
    print(f"Per-class Accuracy: {[correct_per_class[i] / total_per_class[i] if total_per_class[i] > 0 else 0 for i in range(len(correct_per_class))]}")
    print(f"Average Accuracy (AA): {aa:.4f}")
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}, Validation AA: {aa:.4f}")
    return avg_loss, accuracy, aa

@torch.no_grad()
def evalute(model, dataloader, device, class_names=None, save_path=None):
    model.eval()

    all_preds = []
    all_labels = []
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    kappa = cohen_kappa_score(all_labels, all_preds)

    print("Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Cohen's Kappa: {kappa:.4f}")

    # ===== 混淆矩阵 =====
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure()
    plt.imshow(cm)
    plt.colorbar()
    
    if class_names is not None:
        plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
        plt.yticks(np.arange(len(class_names)), class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # 在格子中显示数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

    return accuracy, precision, recall, f1

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train AudioCNN1D on Qiandao Dataset")
    parser.add_argument('--model_name', type=str, default='audiocnn1d', help='Model name')
    parser.add_argument('--dataset', type=str, default='Deepship', help='Dataset name')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or evaluate')
    parser.add_argument('--train_data_json', type=str, default='/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_train_data.json', help='Path to dataset JSON file')
    parser.add_argument('--eval_data_json', type=str, default='/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_eval_data.json', help='Path to evaluation dataset JSON file')
    parser.add_argument('--label_csv', type=str, default='/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_class_map.csv', help='Path to label CSV file')  
    parser.add_argument('--model_path', type=str, default='', help='Path to the trained model for evaluation')
    parser.add_argument('--classes', type=int, default=4, help='Number of classes for training and evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--transform', type=str, default='fft', help='Feature transform: fft or mel')
    parser.add_argument('--sr', type=int, default=52734, help='Sample rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--ft_entire_network', type=bool, default=False, help='Whether to fine-tune the entire network')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze WavLM backbone during training')
    parser.add_argument('--pool_mode', type=str, default='mean', choices=['mean','max','attention'], help='Pooling mode for WavLM classifier')
    parser.add_argument('--attention_hidden', type=int, default=128, help='Hidden dim for attention pooling (if used)')
    parser.add_argument('--pool_dropout', type=float, default=0.1, help='Dropout applied before classifier pooling')
    parser.add_argument('--ration', type=float, default=0.0, help='Ratio for data splitting')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    batch_size = args.batch_size
    transform = args.transform
    sr = args.sr
    lr = args.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset_name = args.dataset
    model_name = args.model_name
    if model_name.lower() == 'audiocnn1d':
        model = AudioCNN1D(num_classes=args.classes).to(device)
    elif model_name.lower() == 'audiocnn2d':
        model = AudioCNN2D(num_classes=args.classes).to(device)
    elif model_name.lower() == 'audiolstm':
        model = AudioLSTM(num_classes=args.classes).to(device)
    elif model_name.lower() == 'resnetaudio':
        model = ResNetAudio(num_classes=args.classes).to(device)
    elif model_name.lower() == 'ast':
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
    elif model_name.lower() == 'wavlm':
        model = WavLMClassifier(
            num_classes=args.classes,
            freeze_backbone=args.freeze_backbone,
            pool_mode=args.pool_mode,
            attention_hidden=args.attention_hidden,
            dropout=args.pool_dropout,
        ).to(device)
    elif model_name.lower() == 'beats':
        model = BEATsTransferLearningModel(
            num_target_classes=args.classes,
            ft_entire_network=args.ft_entire_network
        )

        model.to(device)
        if model.ft_entire_network:
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.beats.parameters()},
                    {"params": model.fc.parameters()}
                ],
                lr=lr,
                betas=(0.9, 0.98),
                weight_decay=1e-5
            )
        else:
            optimizer = torch.optim.AdamW(
                model.fc.parameters(),
                lr=lr,
                betas=(0.9, 0.98),
                weight_decay=1e-5
    )
    
    if model_name.lower() != 'beats':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(params,
                                     lr=lr,
                                     weight_decay=1e-5
                                     )
    
    # 添加学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=lr * 0.1)
    
    criterion = nn.CrossEntropyLoss()
    print("使用的模型为：", model_name, "数据集分割比例为：", args.ration)
    if args.mode == 'evaluate':
        val_data_path = args.eval_data_json
        label_csv_path = args.label_csv
        test_dataset = AudioDataset(dataset_json_file=val_data_path, label_csv_file=label_csv_path, n_fft=8192, transform=transform, sr=sr)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model_path = args.model_path  # 修改为实际模型路径
        print(f"Loading model from {model_path} for evaluation...")
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        if args.classes == 4:
            class_names = ['Cargo', 'Passengership', 'Tanker', 'Tug']
        else:
            class_names = None
        evalute(model, test_loader, device, class_names=class_names, save_path=f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/{model_name}_confusion_matrix.png")
        exit(0)
    else:

        train_data_path = args.train_data_json
        val_data_path = args.eval_data_json
        label_csv_path = args.label_csv

        train_dataset = AudioDataset(dataset_json_file=train_data_path, label_csv_file=label_csv_path, n_fft=8192, transform=transform, sr=sr, ration=args.ration, train=True)
        val_dataset = AudioDataset(dataset_json_file=val_data_path, label_csv_file=label_csv_path, n_fft=8192, transform=transform, sr=sr, ration=args.ration, train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        num_epochs = args.num_epochs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if args.ration > 0.0:
            log_file = f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}_{args.ration}/logs/{model_name}_log_{timestamp}.txt"
            os.makedirs(f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}_{args.ration}/logs", exist_ok=True)
            os.makedirs(f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}_{args.ration}/ckpt", exist_ok=True)
            writer = SummaryWriter(f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}_{args.ration}/tensorboard/{model_name}_{timestamp}")
        else:
            log_file = f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/logs/{model_name}_log_{timestamp}.txt"
            os.makedirs(f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/logs", exist_ok=True)
            os.makedirs(f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/ckpt", exist_ok=True)
            writer = SummaryWriter(f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/tensorboard/{model_name}_{timestamp}")
        global_step = 0
        best_acc = 0.0
        args_dict = vars(args)

        with open(log_file, "w") as f:
            f.write("="*60 + "\n")
            f.write("Experiment Configuration\n")
            f.write("="*60 + "\n")
            f.write(json.dumps(args_dict, indent=4))
            f.write("\n" + "="*60 + "\n\n")
        for epoch in range(num_epochs):
            train_loss, train_acc, global_step = train_one_epoch(model, train_loader, criterion, optimizer, device, global_step, writer)
            val_loss, val_acc, val_aa = validate(model, val_loader, criterion, device)
            log_str = (f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AA: {val_aa:.4f}")
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('AA/val', val_aa, epoch)
            
            # 学习率调度器步进
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning Rate', current_lr, epoch)
            
            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")
            if args.ration > 0.0:
                save_path = f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}_{args.ration}/ckpt/{model_name}_best.pth"
            else:
                save_path = f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/ckpt/{model_name}_best.pth"
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
            if args.ration > 0.0:
                save_path = f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}_{args.ration}/ckpt/{model_name}_AA{val_aa:.4f}.pth"
            else:
                save_path = f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/ckpt/{model_name}_AA{val_aa:.4f}.pth"
            # torch.save(model.state_dict(), save_path)
        writer.close()
        save_path = f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/ckpt/{model_name}_{timestamp}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
