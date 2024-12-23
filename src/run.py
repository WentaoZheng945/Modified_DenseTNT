import argparse
import itertools
import logging
import os
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm as tqdm_

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 记录警告和错误


def compile_pyx_files():
    """检查是否编译了pyx文件，或者是否修改了pyx，需要重新编译"""
    if True:
        os.chdir('src/')
        if not os.path.exists('utils_cython.c') or \
                os.path.getmtime('utils_cython.pyx') > os.path.getmtime('utils_cython.c'):
            os.system('cython -a utils_cython.pyx && python setup.py build_ext --inplace')
        os.chdir('../')


# Comment out this line if pyx files have been compiled manually.
compile_pyx_files()

import utils, structs
from modeling.vectornet import VectorNet

logging.basicConfig(# filename='run.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm = partial(tqdm_, dynamic_ncols=True)


def is_main_device(device):
    return isinstance(device, torch.device) or device == 0


def learning_rate_decay(args, i_epoch, optimizer, optimizer_2=None):
    """"""
    utils.i_epoch = i_epoch

    if 'set_predict' in args.other_params:
        if not hasattr(args, 'set_predict_lr'):
            args.set_predict_lr = 1.0
        else:
            args.set_predict_lr *= 0.9

        if i_epoch > 0 and i_epoch % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.3

        if 'complete_traj-3' in args.other_params:
            assert False

    else:
        if i_epoch > 0 and i_epoch % 5 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.3

        if 'complete_traj-3' in args.other_params:
            if i_epoch > 0 and i_epoch % 5 == 0:
                for p in optimizer_2.param_groups:
                    p['lr'] *= 0.3


def gather_and_output_motion_metrics(args, device, queue, motion_metrics, metric_names, MotionMetrics):
    if is_main_device(device):
        for i in range(args.distributed_training - 1):
            motion_metrics_ = queue.get()
            assert isinstance(motion_metrics_, MotionMetrics), type(motion_metrics_)
            for each in zip(*motion_metrics_.get_all()):
                motion_metrics.update_state(*each)
        print('all metric_values', len(motion_metrics.get_all()[0]))

        score_file = utils.get_eval_identifier()

        utils.logging(utils.metric_values_to_string(motion_metrics.result(), metric_names),
                      type=score_file, to_screen=True, append_time=True)

    else:
        queue.put(motion_metrics)


def gather_and_output_others(args, device, queue, motion_metrics):
    if is_main_device(device):
        for i in range(args.distributed_training - 1):
            other_errors_dict_ = queue.get()
            for key in utils.other_errors_dict:
                utils.other_errors_dict[key].extend(other_errors_dict_[key])

        score_file = score_file = utils.get_eval_identifier()
        utils.logging('other_errors {}'.format(utils.other_errors_to_string()),
                      type=score_file, to_screen=True, append_time=True)

    else:
        queue.put(utils.other_errors_dict)


def single2joint(pred_trajectory, pred_score, args):
    assert pred_trajectory.shape == (2, 6, args.future_frame_num, 2)
    assert np.all(pred_score < 0)
    pred_score = np.exp(pred_score)
    li = []
    scores = []
    for i in range(6):
        for j in range(6):
            score = pred_score[0, i] * pred_score[0, j]
            scores.append(score)
            li.append((score, i, j))

    argsort = np.argsort(-np.array(scores))

    pred_trajectory_joint = np.zeros((6, 2, args.future_frame_num, 2))
    pred_score_joint = np.zeros(6)

    for k in range(6):
        score, i, j = li[argsort[k]]
        pred_trajectory_joint[k, 0], pred_trajectory_joint[k, 1] = pred_trajectory[0, i], pred_trajectory[1, j]
        pred_score_joint[k] = score

    return np.array(pred_trajectory_joint), np.array(pred_score_joint)


def pair2joint(pred_trajectory, pred_score, args):
    assert pred_trajectory.shape == (2, 6, args.future_frame_num, 2)

    pred_trajectory_joint = np.zeros((6, 2, args.future_frame_num, 2))
    pred_score_joint = np.zeros(6)
    for k in range(6):
        assert utils.equal(pred_score[0, k], pred_score[1, k])
        pred_trajectory_joint[k, 0] = pred_trajectory[0, k]
        pred_trajectory_joint[k, 1] = pred_trajectory[1, k]
        pred_score_joint[k] = pred_score[0, k]

    return pred_trajectory_joint, pred_score_joint


def train_one_epoch(model, iter_bar, optimizer, device, args: utils.Args, i_epoch, queue=None, optimizer_2=None):
    """
    训练一个epoch
    Args:
        model: VectorNet
        iter_bar:进度条
        optimizer:优化器一
        device:GPU编号
        args:参数
        i_epoch:第i个epoch
        queue:共享队列
        optimizer_2:优化器二

    Returns:

    """
    li_ADE = []  # 平均位移误差
    li_FDE = []  # 最终位移误差
    utils.other_errors_dict.clear()  # 置空
    start_time = time.time()  # 开始时间
    if 'data_ratio_per_epoch' in args.other_params:  # 没执行
        max_iter_num = int(float(args.other_params['data_ratio_per_epoch']) * len(iter_bar))
        if is_main_device(device):
            print('data_ratio_per_epoch', float(args.other_params['data_ratio_per_epoch']))

    if args.distributed_training:
        assert dist.get_world_size() == args.distributed_training  # gpu个数

    for step, batch in enumerate(iter_bar):
        if 'data_ratio_per_epoch' in args.other_params:
            max_iter_num -= 1
            if max_iter_num == 0:
                break
        loss, DE, _ = model(batch, device)  # 计算loss和位移误差DE
        loss.backward()  # 反向传播

        if is_main_device(device):
            iter_bar.set_description(f'loss={loss.item():.3f}')  # 设置进度条

        final_idx = batch[0].get('final_idx', -1)
        li_FDE.extend([each for each in DE[:, final_idx]])

        if optimizer_2 is not None:
            optimizer_2.step()  # 更新优化器
            optimizer_2.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

    if not args.debug and is_main_device(device):
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(
            args.model_save_dir, "model.{0}.bin".format(i_epoch + 1))
        torch.save(model_to_save.state_dict(), output_model_file)  # 保存模型

    if args.argoverse:
        if is_main_device(device):
            for i in range(args.distributed_training - 1):
                other_errors_dict_ = queue.get()
                for key in utils.other_errors_dict:
                    utils.other_errors_dict[key].extend(other_errors_dict_[key])  # 保存错误
        else:
            queue.put(utils.other_errors_dict)

    if is_main_device(device):
        print()
        miss_rates = (utils.get_miss_rate(li_FDE, dis=2.0), utils.get_miss_rate(li_FDE, dis=4.0),
                      utils.get_miss_rate(li_FDE, dis=6.0))

        utils.logging(f'FDE: {np.mean(li_FDE) if len(li_FDE) > 0 else None}',
                      f'MR(2m,4m,6m): {miss_rates}',
                      type='train_loss', to_screen=True)


def run_training_process(rank, world_size, args, queue):
    if args.distributed_training:
        print(f"Running DDP on rank {rank}.")

        def setup(rank, world_size):
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = args.master_port

            # initialize the process group
            dist.init_process_group("nccl", rank=rank, world_size=world_size)

        setup(rank, world_size)

        utils.args = args
        model = VectorNet(args).to(rank)

        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = VectorNet(args).to(rank)

    if 'set_predict' in args.other_params:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    elif 'complete_traj-3' in args.other_params:
        prefix = 'module.' if isinstance(model, DDP) else ''
        optimizer = torch.optim.Adam(
            [each[1] for each in model.named_parameters() if not str(each[0]).startswith(f'{prefix}decoder.complete_traj')],
            lr=args.learning_rate)  # 一个用于优化不以decoder.complete_traj开头的参数
        optimizer_2 = torch.optim.Adam(
            [each[1] for each in model.named_parameters() if str(each[0]).startswith(f'{prefix}decoder.complete_traj')],
            lr=args.learning_rate)  # 另一个用于优化以decoder.complete_traj开头的参数
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.distributed_training:
        if rank == 0:
            receive = queue.get()
            assert receive == True
        dist.barrier()
        args.reuse_temp_file = True

    if args.argoverse:
        from dataset_argoverse import Dataset, DatasetTrain
        train_dataset = Dataset(args, args.train_batch_size, to_screen=args.distributed_training == 0)

        if args.distributed_training:
            train_sampler = DistributedSampler(train_dataset, shuffle=args.do_train)
        else:
            train_sampler = RandomSampler(train_dataset)

        assert args.train_batch_size == 64, 'The optimal total batch size for training is 64'
        assert args.train_batch_size % world_size == 0
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=args.train_batch_size // world_size,
            collate_fn=utils.batch_list_to_batch_tensors)  # 保证每个进程处理的batch_size=total_batch_size//进程数

    for i_epoch in range(int(args.num_train_epochs)):
        if 'complete_traj-3' in args.other_params:
            learning_rate_decay(args, i_epoch, optimizer, optimizer_2)
        else:
            learning_rate_decay(args, i_epoch, optimizer)
        utils.logging(optimizer.state_dict()['param_groups'])
        if rank == 0:
            print('Epoch: {}/{}'.format(i_epoch, int(args.num_train_epochs)), end='  ')
            print('Learning Rate = %5.8f' % optimizer.state_dict()['param_groups'][0]['lr'])
        if args.distributed_training:
            train_sampler.set_epoch(i_epoch)
        if rank == 0:
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        else:
            iter_bar = train_dataloader

        if 'complete_traj-3' in args.other_params:
            train_one_epoch(model, iter_bar, optimizer, rank, args, i_epoch, queue, optimizer_2)
        else:
            train_one_epoch(model, iter_bar, optimizer, rank, args, i_epoch, queue)

        if args.distributed_training:
            dist.barrier()
    if args.distributed_training:
        dist.destroy_process_group()


def do_train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("Loading dataset", args.data_dir)
    if args.argoverse:
        from dataset_argoverse import Dataset, DatasetTrain

    if args.distributed_training:
        queue = mp.Manager().Queue()  # 共享队列
        # initialize dataset before distributed training
        print("\033[31m" + 'Init loading dataset' + "\033[0m")
        Dataset(args, args.train_batch_size)
        print("\033[31m" + 'Finish loading dataset' + "\033[0m")
        print("\033[31m" + 'Finish loading dataset' + "\033[0m")
        spawn_context = mp.spawn(run_training_process,
                                 args=(args.distributed_training, args, queue),
                                 nprocs=args.distributed_training,
                                 join=False)
        queue.put(True)
        while not spawn_context.join():  # 一直执行直到子进程全部执行结束
            pass
    else:
        run_training_process(0, 1, args, None)


def main():
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args, logger)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    print("\033[31m" + 'Finish init training configuration' + "\033[0m")

    if args.argoverse:
        if args.do_train:
            do_train(args)
        else:
            from do_eval import do_eval
            do_eval(args)
    else:
        assert False

    logger.info('Finish.')


if __name__ == "__main__":
    main()
