#!/bin/sh
# ---------------------------------------------
# Bash wrapper for AudioCNN1D training/evaluation
# ---------------------------------------------

# 设置 Python 脚本路径
PYTHON_SCRIPT="/data/zcx/wav_prj/Qiandao/src/train.py"

# 默认参数
MODE="train"   # train / evaluate
DATASET="Deepship"
TRAIN_JSON="/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_train_data.json"
EVAL_JSON="/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_eval_data.json"
LABEL_CSV="/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_class_map.csv"
MODEL_PATH=""  # 如果 mode=evaluate，需要填路径
CLASSES=4
BATCH_SIZE=32
MODEL_NAME="ResNetAudio"
SR=16000
TRANSFORM="fbank"
LR=0.001
EPOCHS=20



# ------------------------
# 执行 Python 脚本
# ------------------------
python3 "${PYTHON_SCRIPT}" \
  --dataset "${DATASET}" \
  --mode "${MODE}" \
  --train_data_json "${TRAIN_JSON}" \
  --eval_data_json "${EVAL_JSON}" \
  --label_csv "${LABEL_CSV}" \
  --model_path "${MODEL_PATH}" \
  --classes "${CLASSES}"\
  --batch_size "${BATCH_SIZE}"\
  --model_name "${MODEL_NAME}"\
  --sr "${SR}"\
  --transform "${TRANSFORM}"\
  --lr "${LR}"\
  --num_epochs "${EPOCHS}"
