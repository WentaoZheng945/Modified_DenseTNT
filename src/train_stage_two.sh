OUTPUT_DIR=argoverse2.densetnt.set_predict.1; \
MODEL_PATH=argoverse2.densetnt.1/model_save/model.16.bin; \
GPU_NUM=4; \
python src/run.py --argoverse --argoverse2 --future_frame_num 60 \
  --do_train --data_dir /root/autodl-tmp/data/train --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 48 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj \
    set_predict=6 set_predict-6 data_ratio_per_epoch=0.4 set_predict-topk=0 set_predict-one_encoder set_predict-MRratio=0.0 \
    set_predict-train_recover=${MODEL_PATH} \