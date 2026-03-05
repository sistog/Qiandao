#!/bin/sh
# ---------------------------------------------
# Bash wrapper for AudioCNN1D training/evaluation
# ---------------------------------------------

# 设置 Python 脚本路径
PYTHON_SCRIPT="/data/zcx/wav_prj/Qiandao/src/train.py"

# 默认参数
MODE="train"   # train / evaluate
DATASET="UUV"
TRAIN_JSON="/data/zcx/wav_prj/Qiandao/src/datafiles/uuv_train_data.json"
EVAL_JSON="/data/zcx/wav_prj/Qiandao/src/datafiles/uuv_eval_data.json"
LABEL_CSV="/data/zcx/wav_prj/Qiandao/src/datafiles/uuv_class_map.csv"
MODEL_PATH=""  # 如果 mode=evaluate，需要填路径
CLASSES=2
BATCH_SIZE=4
MODEL_NAME="Beats"
SR=52734
TRANSFORM="raw"
LR=5e-5
EPOCHS=20
FT_ENTIRE_NETWORK=False



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
  --num_epochs "${EPOCHS}"\
  --ft_entire_network FT_ENTIRE_NETWORK
