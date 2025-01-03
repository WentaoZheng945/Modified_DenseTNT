import argparse
import logging
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm

import structs
import utils
from modeling.vectornet import VectorNet

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = partial(tqdm, dynamic_ncols=True)


def eval_instance_argoverse(batch_size, args, pred, mapping, file2pred, file2labels, DEs, iter_bar):
    for i in range(batch_size):
        a_pred = pred[i]
        assert a_pred.shape == (6, args.future_frame_num, 2)

        if args.argoverse2:
            file_name = os.path.split(mapping[i]['file_name'])[1]
            file2pred[file_name] = a_pred
        else:
            file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
            file2pred[file_name_int] = a_pred

            if not args.do_test:
                file2labels[file_name_int] = mapping[i]['origin_labels']

    if not args.do_test:
        DE = np.zeros([batch_size, args.future_frame_num])
        for i in range(batch_size):
            origin_labels = mapping[i]['origin_labels']
            for j in range(args.future_frame_num):
                DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                        origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
        DEs.append(DE)
        miss_rate = 0.0
        if 0 in utils.method2FDEs:
            FDEs = utils.method2FDEs[0]
            miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)

        iter_bar.set_description('Iter (MR=%5.3f)' % (miss_rate))


def do_eval(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Loading Evalute Dataset", args.data_dir)
    if args.argoverse:
        from dataset_argoverse import Dataset,DatasetEval
    if args.eval_single:
        eval_dataset = Dataset(args, args.eval_batch_size)
    else:
        eval_dataset = DatasetEval(args, args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)  # 顺序遍历，不需要分布式训练
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                                                  sampler=eval_sampler,
                                                  collate_fn=utils.batch_list_to_batch_tensors,  # 对版本数据的格式进行处理
                                                  pin_memory=False)
    model = VectorNet(args)
    print('torch.cuda.device_count', torch.cuda.device_count())  # 查看当前系统中有多少可用的GPU

    logger.info("***** Recover model: %s *****", args.model_recover_path)
    if args.model_recover_path is None:
        raise ValueError("model_recover_path not specified.")

    model_recover = torch.load(args.model_recover_path)  # 加载训练好的模型参数
    model.load_state_dict(model_recover)

    if 'set_predict-train_recover' in args.other_params and 'complete_traj' in args.other_params:
        model_recover = torch.load(args.other_params['set_predict-train_recover'])
        utils.load_model(model.decoder.complete_traj_cross_attention, model_recover, prefix='decoder.complete_traj_cross_attention.')
        utils.load_model(model.decoder.complete_traj_decoder, model_recover, prefix='decoder.complete_traj_decoder.')

    model.to(device)
    model.eval()
    file2pred = {}  # 预测结果
    file2labels = {}  # 标签
    iter_bar = tqdm(eval_dataloader, desc='Iter (loss=X.XXX)')  # 进度条
    DEs = []  # 位移偏差
    length = len(iter_bar)

    if args.argoverse2:
        metrics = utils.PredictionMetrics()

    argo_pred = structs.ArgoPred()
    if args.eval_single:
        record = {}
    else:
        record = {}


    for step, batch in enumerate(iter_bar):
        pred_trajectory, pred_score, _ = model(batch, device)  # pre_traject为ndarray，shape为(batch_size, 6, 60, 2) pred_score为ndarray, shape为(batch_size, 6)

        mapping = batch
        batch_size = pred_trajectory.shape[0]
        for i in range(batch_size):
            assert pred_trajectory[i].shape == (6, args.future_frame_num, 2)
            assert pred_score[i].shape == (6,)
            argo_pred[mapping[i]['file_name']] = structs.MultiScoredTrajectory(pred_score[i].copy(), pred_trajectory[i].copy())  # 储存一下每个样本的预测结果和得分

        if args.argoverse2:
            for i in range(batch_size):
                from av2.datasets.motion_forecasting.eval.metrics \
                    import compute_ade, compute_fde, compute_brier_fde

                # 记录预测的轨迹
                # record[batch[i]['file_name']] = pred_trajectory[i]
                if args.eval_single:
                    record[batch[i]['file_name']] = pred_trajectory[i]
                else:
                    record[step*args.eval_batch_size+i] = batch[i]

                forecasted_trajectories = pred_trajectory[i][:, :, :]
                gt_trajectory = mapping[i]['gt_trajectory_global_coordinates'][:, :]
                forecast_probabilities = np.exp(pred_score[i])
                forecast_probabilities = forecast_probabilities * (1.0 / forecast_probabilities.sum())

                assert forecasted_trajectories.shape == (6, 60, 2)
                assert gt_trajectory.shape == (60, 2)
                assert forecast_probabilities.shape == (6,)

                ade = compute_ade(forecasted_trajectories, gt_trajectory)  # shape(k, )
                fde_1 = compute_fde(forecasted_trajectories[:, :10, :], gt_trajectory[:10, :])
                fde_3 = compute_fde(forecasted_trajectories[:, :30, :], gt_trajectory[:30, :])
                fde = compute_fde(forecasted_trajectories, gt_trajectory)  # shape(k, )
                brier_fde = compute_brier_fde(forecasted_trajectories, gt_trajectory, forecast_probabilities)  # shape(k, )
                metrics.minADE.accumulate(ade.min())
                metrics.minFDE_1.accumulate(fde_1.min())
                metrics.minFDE_3.accumulate(fde_3.min())
                metrics.minFDE.accumulate(fde.min())
                metrics.MR.accumulate(fde.min() > 2.0)
                metrics.brier_minFDE.accumulate(brier_fde.min())
        else:
            eval_instance_argoverse(batch_size, args, pred_trajectory, mapping, file2pred, file2labels, DEs, iter_bar)

    if 'optimization' in args.other_params:
        utils.select_goals_by_optimization(None, None, close=True)

    # 记录当前结果
    import pickle
    if args.eval_single:
        # pickle_file = open(os.path.join(args.temp_file_dir, 'predict_result'), 'wb')
        pickle_file = open(os.path.join('/root/autodl-tmp/data/preprocess_data/', 'predict_result'), 'wb')
    else:
        # pickle_file = open(os.path.join(args.temp_file_dir, 'full_info'), 'wb')
        pickle_file = open(os.path.join('/root/autodl-tmp/data/preprocess_data/', 'full_info5'), 'wb')
    pickle.dump(record, pickle_file)
    pickle_file.close()

    if args.argoverse2:
        import json
        print('Metrics:')
        print(json.dumps(metrics.serialize(), indent=4))
    else:
        from dataset_argoverse import post_eval
        post_eval(args, file2pred, file2labels, DEs)


def main():
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args, logger)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    do_eval(args)


if __name__ == "__main__":
    main()
