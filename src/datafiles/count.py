import json

train_path = '/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_train_data.json'
eval_path = '/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_eval_data.json'

def count_samples(json_path):
    with open(json_path, 'r') as fp:
        data_json = json.load(fp)
    data_list = data_json['data']
    return len(data_list)

if __name__ == "__main__":
    train_count = count_samples(train_path)
    eval_count = count_samples(eval_path)
    print(f"Train samples: {train_count}")
    print(f"Eval samples: {eval_count}")
