import pandas as pd
import json
import os
from pathlib import Path
from collections import defaultdict

file_path = "/data/zcx/wav_prj/Qiandao/src/dataset/esc-50/ESC-50-master/meta/esc50.csv"

ma = defaultdict(int)

def create_json_from_csv(csv_file, output_train_json, output_val_json, map_csv_file):
    df = pd.read_csv(csv_file)
    train_data = []
    val_data = []
    for _, row in df.iterrows():
        item = {
            "wav": os.path.join("/data/zcx/wav_prj/Qiandao/src/dataset/esc-50/ESC-50-master/audio_32k", row['filename']),
            "labels": row['category'],
        }
        fold = row['fold']
        if fold in [1, 2, 3, 4]:  # 前4折作为训练集
            train_data.append(item)
        else:  # 第5折作为验证集
            val_data.append(item)
    
    train_json = {"data": train_data}
    val_json = {"data": val_data}
    
    with open(output_train_json, 'w') as f:
        json.dump(train_json, f, indent=4)
    
    with open(output_val_json, 'w') as f:
        json.dump(val_json, f, indent=4)

    with open(map_csv_file, 'w') as f:
        f.write("index,mid,display_name\n")
        for idx, category in enumerate(sorted(df['category'].unique())):
            f.write(f"{idx},{category},{category}\n")



if __name__ == "__main__":
    create_json_from_csv(
        csv_file=file_path,
        output_train_json="/data/zcx/wav_prj/Qiandao/src/datafiles/esc50_train_data.json",
        output_val_json="/data/zcx/wav_prj/Qiandao/src/datafiles/esc50_val_data.json",
        map_csv_file = "/data/zcx/wav_prj/Qiandao/src/datafiles/esc50_labels_map.csv"
    )