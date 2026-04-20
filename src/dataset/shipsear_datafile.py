from pathlib import Path
from typing import List, Tuple
import glob
import json
import os
import csv  

TRAIN_DIR = "/data/zcx/wav_prj/Qiandao/src/dataset/shipsear_segmented/train"
TEST_DIR = "/data/zcx/wav_prj/Qiandao/src/dataset/shipsear_segmented/test"

types = set()

def get_file_list(root_dir: str, extensions: List[str]) -> List[Path]:
    """
    获取指定目录下所有具有特定扩展名的文件列表。
    
    Args:
        root_dir (str): 根目录路径。
        extensions (List[str]): 需要匹配的文件扩展名列表（例如 ['.wav', '.mp3']）。 
    """
    file_list = []
    
    for ext in extensions:
        for file in Path(root_dir).rglob(f'*{ext}'):
            label = str(file).split('/')[-1].split('_')[0]  # 根据文件命名规则提取标签
            d = {}
            label = label.lower()
            d['wav'] = str(file)
            d['labels'] = label
            types.add(label)
            file_list.append(d)
    return file_list
train_files = get_file_list(TRAIN_DIR, ['.wav'])
data = {'data': train_files}
with open('/data/zcx/wav_prj/Qiandao/src/datafiles/shipsear_train_data.json', 'w') as f:
    json.dump(data, f, indent=4)
test_files = get_file_list(TEST_DIR, ['.wav'])
data = {'data': test_files}
with open('/data/zcx/wav_prj/Qiandao/src/datafiles/shipsear_val_data.json', 'w') as f:
    json.dump(data, f, indent=4)
print(f"✅ 训练集文件数量: {len(train_files)}")
print(f"✅ 测试集文件数量: {len(test_files)}")
print(f"✅ 标签类别: {types}")

csv_file = "/data/zcx/wav_prj/Qiandao/src/datafiles/shipsear_label.csv"
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = ['mid', 'index', 'display_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for idx, label in enumerate(sorted(types)):
        writer.writerow({'index': idx,'mid': label, 'display_name': label})