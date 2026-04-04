#!/bin/sh
# ---------------------------------------------
# Bash wrapper for AudioCNN1D training/evaluation
# ---------------------------------------------

# 激活包含PyTorch的环境
source /usr/local/conda/bin/activate /data/zcx/conda_envs/ast_env

# 设置 Python 脚本路径
PYTHON_SCRIPT="/data/zcx/wav_prj/Qiandao/src/train.py"

# 默认参数
MODE="train"   # train / evaluate
DATASET="Deepship"
TRAIN_JSON="/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_train_data.json"
EVAL_JSON="/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_eval_data.json"
LABEL_CSV="/data/zcx/wav_prj/Qiandao/src/datafiles/deepship_class_map.csv"
MODEL_PATH="/data/zcx/wav_prj/Qiandao/src/exp/Deepship/ckpt/Beats_best.pth"  # 如果 mode=evaluate，需要填路径
CLASSES=4
BATCH_SIZE=16
MODEL_NAME="beats"
SR=32000
TRANSFORM="raw"
LR=2e-5
EPOCHS=50
FT_ENTIRE_NETWORK=True
RATION=0.0  # 数据划分比例，0.0 表示不划分



# ------------------------
# 执行 Python 脚本
# ------------------------
# for r in {0.2,0.3,0.4,0.5,0.6,0.7,0.8}; do
#   python3 "${PYTHON_SCRIPT}" \
#     --dataset "${DATASET}" \
#     --mode "${MODE}" \
#     --train_data_json "${TRAIN_JSON}" \
#     --eval_data_json "${EVAL_JSON}" \
#     --label_csv "${LABEL_CSV}" \
#     --model_path "${MODEL_PATH}" \
#     --classes "${CLASSES}"\
#     --batch_size "${BATCH_SIZE}"\
#     --model_name "${MODEL_NAME}"\
#     --sr "${SR}"\
#     --transform "${TRANSFORM}"\
#     --lr "${LR}"\
#     --num_epochs "${EPOCHS}"\
#     --ft_entire_network FT_ENTIRE_NETWORK\
#     --ration "$r"
# done
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
  --ft_entire_network FT_ENTIRE_NETWORK\
  --ration "${RATION}"
