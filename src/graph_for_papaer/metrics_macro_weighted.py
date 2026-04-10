import numpy as np

# 1. 原始实验数据
data = {
    'Beats': [[1885, 201, 465, 126], [140, 2858, 410, 381], [368, 341, 2855, 23], [144, 513, 97, 2596]],
    'Transformer': [[1938, 210, 401, 128], [275, 2822, 402, 290], [552, 198, 2605, 232], [132, 214, 515, 2489]],
    'ResNet': [[1741, 177, 594, 165], [149, 2679, 452, 509], [539, 425, 2606, 17], [133, 561, 128, 2528]],
    'CNN': [[1890, 119, 536, 132], [264, 2612, 585, 328], [528, 249, 2793, 17], [333, 717, 235, 2065]],
    'LSTM': [[1796, 109, 637, 135], [348, 2523, 606, 312], [506, 342, 2730, 9], [326, 761, 356, 1907]]
}

def evaluate_model_full(matrix):
    cm = np.array(matrix)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1)
    total_samples = np.sum(cm)
    
    # 计算 Overall Accuracy
    accuracy = np.sum(tp) / total_samples
    
    # 计算各类别基础指标 (处理除零)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
    
    # Macro Average
    macro = [np.mean(precision), np.mean(recall), np.mean(f1)]
    
    # Weighted Average
    weighted = [
        np.sum(precision * support) / total_samples,
        np.sum(recall * support) / total_samples,
        np.sum(f1 * support) / total_samples
    ]
    
    return accuracy, macro, weighted

# 2. 打印学术对比表
print(f"{'Model':<15} | {'Acc':<8} | {'Macro (P/R/F1)':<28} | {'Weighted (P/R/F1)':<28}")
print("-" * 100)

for name, matrix in data.items():
    acc, ma, we = evaluate_model_full(matrix)
    
    print(f"{name:<15} | {acc:.4f}   | "
          f"{ma[0]:.4f} / {ma[1]:.4f} / {ma[2]:.4f} | "
          f"{we[0]:.4f} / {we[1]:.4f} / {we[2]:.4f}")