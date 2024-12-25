OUTPUT_DIR=argoverse2.densetnt.1; \
GPU_NUM=4; \
python src/run.py --argoverse --argoverse2 --future_frame_num 60 \
  --do_train --data_dir /root/autodl-tmp/data/train --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 48 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3 \
  --do_eval --eval_params optimization MRminFDE=0.0 cnt_sample=9 opti_time=0.1 \
  --visualize_post
  # --eval_single
  # --visualize --visualize_num 100
  # 添加eval_single 是作用于单车，用于评估各项指标；不添加eval_single 是作用于多车，用于绘图以及生成多种危险场景