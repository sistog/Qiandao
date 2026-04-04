import json
import torchaudio
import torch

TRAIN_JSON="/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_train_data.json"
EVAL_JSON="/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_eval_data.json"

if __name__ == "__main__":
    with open(TRAIN_JSON, 'r') as fp:
        train_data_json = json.load(fp)
    with open(EVAL_JSON, 'r') as fp:
        eval_data_json = json.load(fp)

    sm = len(train_data_json['data'])+len(eval_data_json['data'])
    train_len = len(train_data_json['data'])
    eval_len = len(eval_data_json['data'])
    print(f"Train data count: {train_len}")
    print(f"Eval data count: {eval_len}")
    print(f"Total data count: {sm}")
    print(f"Train data percentage: {train_len/sm:.2%}")
    print(f"Eval data percentage: {eval_len/sm:.2%}")

    # Train data count: 43065
    # Eval data count: 13403
    # Total data count: 56468
    # Train data percentage: 76.26%
    # Eval data percentage: 23.74%

    TRAIN_JSON="/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_train_data.json"

def get_mix_max():
    with open(TRAIN_JSON, 'r') as fp:
        data_json = json.load(fp)
    data_list = data_json['data']
    global_min = float('inf')
    global_max = float('-inf')
    for item in data_list:
        path = item['wav']
        waveform, sr = torchaudio.load(path)
        file_min = torch.min(waveform).item()
        file_max = torch.max(waveform).item()
        if file_min < global_min:
            global_min = file_min
        if file_max > global_max:
            global_max = file_max
    return global_min, global_max

print(get_mix_max())
# (-1.0, 0.93310546875)