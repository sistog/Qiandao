import json
from collections import defaultdict

f = json.load(open('/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_eval_data.json'))
train_data = f['data']
cnt = defaultdict(int)
recording = defaultdict(set)
for each in train_data:
    label = each['labels']
    cnt[label] += 1
    file_path = each['wav']
    recording[label].add(file_path.split('/')[-2])
for label in cnt:
    print(f'{label}: {cnt[label]}')
    print(f'{label} has {len(recording[label])} recordings')
