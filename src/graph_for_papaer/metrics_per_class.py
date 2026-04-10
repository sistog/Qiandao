import numpy as np

# 实验数据
data = {
    'Beats': [[1885, 201, 465, 126], [140, 2858, 410, 381], [368, 341, 2855, 23], [144, 513, 97, 2596]],
    'Transformer': [[1938, 210, 401, 128], [275, 2822, 402, 290], [552, 198, 2605, 232], [132, 214, 515, 2489]],
    'ResNet': [[1741, 177, 594, 165], [149, 2679, 452, 509], [539, 425, 2606, 17], [133, 561, 128, 2528]],
    'CNN': [[1890, 119, 536, 132], [264, 2612, 585, 328], [528, 249, 2793, 17], [333, 717, 235, 2065]],
    'LSTM': [[1796, 109, 637, 135], [348, 2523, 606, 312], [506, 342, 2730, 9], [326, 761, 356, 1907]]
}

class_names = ['Cargo', 'Passenger', 'Tanker', 'Tug']

def get_per_class_metrics(matrix):
    cm = np.array(matrix)
    results = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        results[name] = (p, r, f1)
    return results

# 打印表头
print(f"{'Model':<12} | {'Class':<10} | {'Prec':<8} {'Rec':<8} {'F1':<8}")
print("-" * 55)

for model_name, matrix in data.items():
    metrics = get_per_class_metrics(matrix)
    for cls_name, (p, r, f1) in metrics.items():
        print(f"{model_name:<12} | {cls_name:<10} | {p:.4f}   {r:.4f}   {f1:.4f}")
    print("-" * 55)