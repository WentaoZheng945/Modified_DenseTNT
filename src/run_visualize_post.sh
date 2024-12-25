#!/bin/bash

# 检测 GPU 数量
GPU_COUNT=$(nvidia-smi --list-gpus | grep -c 'GPU')

# 根据 GPU 数量设置目录
if [ "$GPU_COUNT" -gt 1 ]; then
    # 多 GPU 环境
    TEMP_FILE_DIR="/home/zwt/thesis/DenseTNT/trained_model/argoverse2.densetnt.1/temp_file/"
    LOG_DIR="/home/zwt/thesis/DenseTNT/trained_model/argoverse2.densetnt.1/"
else
    # 单 GPU 或无 GPU 环境
    TEMP_FILE_DIR="/root/autodl-tmp/data/preprocess_data/"
    LOG_DIR="/root/zwt/DenseTNT/argoverse2.densetnt.1/"
fi

# 设置可视化数量
VISUALIZE_NUM=100

# 运行 visualize_post.py 脚本
python src/visualize_post.py --visualize_num ${VISUALIZE_NUM} \
  --temp_file_dir ${TEMP_FILE_DIR} \
  --log_dir ${LOG_DIR} \
