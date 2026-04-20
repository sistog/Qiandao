import pandas as pd
import torch
import os
import torchaudio
import torchaudio.transforms as T

# --- 配置路径 ---
FILE_ROOT = "/data/zcx/wav_prj/Qiandao/src/dataset/shipsear/"
xls_file = os.path.join(FILE_ROOT, "shipsEar.xlsx")
split_file = "/data/zcx/wav_prj/Qiandao/src/dataset/shipsear split.xlsx"
Dest_ROOT = "/data/zcx/wav_prj/Qiandao/src/dataset/shipsear_segmented/"

# --- 1. 读取并清理划分列表 ---
# 使用 str 确保读取为字符串，并处理空格
split_df = pd.read_excel(split_file, usecols=["Train", "Test"])

train_list = []
test_list = []

for _, row in split_df.iterrows():
    # 使用 split(',') 后去掉每个元素的空格，并统一补齐两位数（如 '6' -> '06'）
    if pd.notna(row['Train']):
        train_list.extend([s.strip().zfill(2) for s in str(row['Train']).split(',')])
    if pd.notna(row['Test']):
        test_list.extend([s.strip().zfill(2) for s in str(row['Test']).split(',')])

print(f"✅ 训练集 ID 数量: {len(train_list)}")
print(f"✅ 测试集 ID 数量: {len(test_list)}")

# --- 2. 创建目标目录 ---
os.makedirs(os.path.join(Dest_ROOT, 'train'), exist_ok=True)
os.makedirs(os.path.join(Dest_ROOT, 'test'), exist_ok=True)

# --- 3. 读取主数据索引 ---
df = pd.read_excel(xls_file, usecols=["ID", "Filename", "Type"])

# --- 4. 循环处理音频 ---
TARGET_SR = 16000
SEG_LEN_SEC = 3
SEG_SAMPLES = TARGET_SR * SEG_LEN_SEC

print("🚀 开始处理音频切片...")

for idx, row in df.iterrows():
    # 统一 ID 格式用于匹配
    curr_id = str(row['ID']).strip().zfill(2)
    
    # 判断归属
    if curr_id in train_list:
        dest_dir = os.path.join(Dest_ROOT, 'train')
    elif curr_id in test_list:
        dest_dir = os.path.join(Dest_ROOT, 'test')
    else:
        # 如果 ID 不在列表里，跳过
        continue

    file_path = os.path.join(FILE_ROOT, str(row['Filename']))
    
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在，跳过: {file_path}")
        continue

    try:
        # 加载音频 (y 形状: [channels, samples])
        y, sr = torchaudio.load(file_path)

        # 统一转为单声道（水声特征提取通常只需要单通道）
        if y.shape[0] > 1:
            y = torch.mean(y, dim=0, keepdim=True)

        # 重采样到 16kHz
        if sr != TARGET_SR:
            y = torchaudio.functional.resample(y, sr, TARGET_SR)
        
        num_samples = y.shape[1]
        
        # 切片逻辑
        # 使用 y.shape[1] 而不是 len(y)，因为 y 是 Tensor [1, samples]
        count = 0
        for i in range(0, num_samples, SEG_SAMPLES):
            segment = y[:, i : i + SEG_SAMPLES]
            
            # 最后一段如果长度不足，补零 (Padding)
            if segment.shape[1] < SEG_SAMPLES:
                padding = torch.zeros((1, SEG_SAMPLES - segment.shape[1]))
                segment = torch.cat((segment, padding), dim=1)
            
            # 生成文件名：类别_ID_原始文件名_序号.wav
            # 加上原始文件名 base_name 是为了防止同一个 ID 的不同原始文件生成的切片重名覆盖
            base_name = os.path.splitext(str(row['Filename']))[0].replace("/", "_")
            count += 1
            segment_filename = f"{row['Type']}_{curr_id}_{base_name}_seg{count}.wav"
            
            save_path = os.path.join(dest_dir, segment_filename)
            torchaudio.save(save_path, segment, TARGET_SR)

        if idx % 10 == 0:
            print(f"已处理第 {idx} 个文件: {row['Filename']} -> 切成 {count} 段")

    except Exception as e:
        print(f"❌ 处理文件 {row['Filename']} 时出错: {e}")

print(f"✨ 所有任务完成！切片存储在: {Dest_ROOT}")