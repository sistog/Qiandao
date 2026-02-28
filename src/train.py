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
from dataset.qiandao_dataset import AudioDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Train")

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        # tqdm 实时显示
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}"
        )

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

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

            # 实时显示
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}"
            )

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

@torch.no_grad()
def evalute(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=lr,
                                weight_decay=1e-5
                                )
    if args.mode == 'evaluate':
        val_data_path = args.eval_data_json
        label_csv_path = args.label_csv
        test_dataset = AudioDataset(dataset_json_file=val_data_path, label_csv_file=label_csv_path, n_fft=8192, transform=transform, sr=sr)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model_path = '/data/zcx/wav_prj/Qiandao/src/exp/ckpt/audiocnn1d_20260120-122121.pth'  # 修改为实际模型路径
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        evalute(model, test_loader, device)
        exit(0)
    else:

        train_data_path = args.train_data_json
        val_data_path = args.eval_data_json
        label_csv_path = args.label_csv

        train_dataset = AudioDataset(dataset_json_file=train_data_path, label_csv_file=label_csv_path, n_fft=8192, transform=transform, sr=sr)
        val_dataset = AudioDataset(dataset_json_file=val_data_path, label_csv_file=label_csv_path, n_fft=8192, transform=transform, sr=sr)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        num_epochs = args.num_epochs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/logs/{model_name}_log_{timestamp}.txt"
        os.makedirs(f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/logs", exist_ok=True)
        os.makedirs(f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/ckpt", exist_ok=True)
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            log_str = (f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            print(log_str)
            with open(log_file, "a") as f:
                f.write(log_str + "\n")
        save_path = f"/data/zcx/wav_prj/Qiandao/src/exp/{dataset_name}/ckpt/{model_name}_{timestamp}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
