OUTPUT_DIR=argoverse2.densetnt.1; \
GPU_NUM=1; \
python src/run.py --argoverse --argoverse2 --future_frame_num 60 \
  --do_train --data_dir /media/zwt/新加卷/Argoverse_2/train/ --output_dir ${OUTPUT_DIR} \
  --hidden_size 128 --train_batch_size 64 --use_map \
  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} \
  --other_params \
    semantic_lane direction l1_loss \
    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
    lane_scoring complete_traj complete_traj-3 \
#python zwt/args.py --argoverse --argoverse2 --future_frame_num 60 \
#  --do_train --data_dir data/train/ --output_dir ${OUTPUT_DIR} \
#  --hidden_size 128 --train_batch_size 64 --use_map \
#  --core_num 16 --use_centerline --distributed_training ${GPU_NUM} \
#  --other_params \
#    semantic_lane direction l1_loss \
#    goals_2D enhance_global_graph subdivide goal_scoring laneGCN point_sub_graph \
#    lane_scoring complete_traj complete_traj-3 \
