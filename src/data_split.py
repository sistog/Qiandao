from pathlib import Path
import os
import json

DATA_DIR = '/data/zcx/wav_prj/PANN_Models_DeepShip-main/Datasets/DeepShip/Segments_3s_16000hz'
CLASSES_NAMES = ['Cargo', 'Passengership', 'Tanker', 'Tug']

def get_id():
    id_list = []
    for x in CLASSES_NAMES:
        dir_path = Path(DATA_DIR) / x
        res = [int(p.name) for p in dir_path.iterdir() if p.is_dir()]
        res.sort()
        id_list.append(res)
    return id_list

def split_data(id_list, train_ratio=0.8):
    train_ids = []
    val_ids = []
    for ids in id_list:
        n_train = int(len(ids) * train_ratio)
        train_ids.append(ids[:n_train])
        val_ids.append(ids[n_train:])
    return train_ids, val_ids

if __name__ == '__main__':
    id_list = get_id()
    for train_ration in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        train_ids, val_ids = split_data(id_list, train_ratio=train_ration)
        # print("Train IDs:", train_ids)
        # print("Validation IDs:", val_ids)
        train_data = []
        val_data = []
        for i, x in enumerate(CLASSES_NAMES):
            
            for id in train_ids[i]:
                dir_path = Path(DATA_DIR) / x / str(id)
                for file in dir_path.iterdir():
                    if file.is_file() and file.suffix == '.wav':
                        train_data.append({"wav": str(file), "labels": x})
            
            for id in val_ids[i]:
                dir_path = Path(DATA_DIR) / x / str(id)
                for file in dir_path.iterdir():
                    if file.is_file() and file.suffix == '.wav':
                        val_data.append({"wav": str(file), "labels": x})
        train_json = {"data": train_data}
        val_json = {"data": val_data}

        os.makedirs('/data/zcx/wav_prj/Qiandao/src/datafiles/ration', exist_ok=True)
        with open(f'/data/zcx/wav_prj/Qiandao/src/datafiles/ration/deepship_train_data_{train_ration}.json', 'w') as f:
            json.dump(train_json, f, indent= 2)
        with open(f'/data/zcx/wav_prj/Qiandao/src/datafiles/ration/deepship_val_data_{train_ration}.json', 'w') as f:
            json.dump(val_json, f, indent= 2)   