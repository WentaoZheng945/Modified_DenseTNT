import argparse

import torch
from modeling.vectornet import VectorNet
from src.utils import add_argument
from typing import Dict


def add_argument(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    # Required parameters
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")  # 是否训练
    parser.add_argument("-e", "--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")  # 是否在开发集上执行评估
    parser.add_argument("--do_test",
                        action='store_true')  # 是否执行测试
    parser.add_argument("--data_dir",
                        default='data/train',
                        type=str)  # 训练数据的目录
    parser.add_argument("--data_dir_for_val",
                        default='data/val',
                        type=str)  # 验证数据的目录
    parser.add_argument("--output_dir", default="tmp/", type=str)  # 输出目录
    parser.add_argument("--log_dir", default=None, type=str)  # 日志目录
    parser.add_argument("--temp_file_dir", default=None, type=str)  # 临时文件目录
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")  # 训练批次的大小

    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")  # 评估批次的大小
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str)  # 模型恢复的路径
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")  # 学习率
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")  # adam优化器的权重衰减率
    parser.add_argument("--num_train_epochs",
                        default=16.0,
                        type=float,
                        help="Total number of training epochs to perform.")  # 训练的总轮数
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")  # 随机种子
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")  # 指使是否不使用cuda
    parser.add_argument("--hidden_size",
                        default=128,
                        type=int)  # 隐藏层大小
    parser.add_argument("--hidden_dropout_prob",
                        default=0.1,
                        type=float)  # dropout的概率
    parser.add_argument("--sub_graph_depth",
                        default=3,
                        type=int)  # 子图的深度
    parser.add_argument("--global_graph_depth",
                        default=1,
                        type=int)  # 全局图的深度
    parser.add_argument("--debug",
                        action='store_true')  # 是否开启调试模式
    parser.add_argument("--initializer_range",
                        default=0.02,
                        type=float)  # 初始化范围
    parser.add_argument("--sub_graph_batch_size",
                        default=8000,
                        type=int)  # 子图批次大小
    parser.add_argument("-d", "--distributed_training",
                        nargs='?',
                        default=8,
                        const=4,
                        type=int)  # 分布式训练参数
    parser.add_argument("--cuda_visible_device_num",
                        default=None,
                        type=int)  # 指定可见的cuda设备数量
    parser.add_argument("--use_map",
                        action='store_true')  # 是否使用映射
    parser.add_argument("--reuse_temp_file",
                        action='store_true')  # 是否重用临时文件
    parser.add_argument("--old_version",
                        action='store_true')  # 是否使用旧版本
    parser.add_argument("--max_distance",
                        default=50.0,
                        type=float)  # 最大距离
    parser.add_argument("--no_sub_graph",
                        action='store_true')  # 是否不使用子图
    parser.add_argument("--no_agents",
                        action='store_true')  # 是否不使用代理
    parser.add_argument("--other_params",
                        nargs='*',
                        default=[],
                        type=str)  # 其他参数
    parser.add_argument("-ep", "--eval_params",
                        nargs='*',
                        default=[],
                        type=str)  # 传递评估参数
    parser.add_argument("-tp", "--train_params",
                        nargs='*',
                        default=[],
                        type=str)  # 传递训练参数
    parser.add_argument("--not_use_api",
                        action='store_true')  # 是否不使用API
    parser.add_argument("--core_num",
                        default=16,
                        type=int)  # 核心数量
    parser.add_argument("--visualize",
                        action='store_true')  # 是否可视化
    parser.add_argument("--train_extra",
                        action='store_true')  # 是否训练额外的模型
    parser.add_argument("--use_centerline",
                        action='store_true')  # 是否使用中心线
    parser.add_argument("--autoregression",
                        nargs='?',
                        default=None,
                        const=2,
                        type=int)  # 设置自回归
    parser.add_argument("--lstm",
                        action='store_true')  # 是否使用lstm
    parser.add_argument("--add_prefix",
                        default=None)  # 是否添加前缀
    parser.add_argument("--attention_decay",
                        action='store_true')  # 是否使用注意力衰减
    parser.add_argument("--placeholder",
                        default=0.0,
                        type=float)  # 占位符
    parser.add_argument("--multi",
                        nargs='?',
                        default=None,
                        const=6,
                        type=int)  # 多任务
    parser.add_argument("--method_span",
                        nargs='*',
                        default=[0, 1],
                        type=int)  # 指定方法跨度
    parser.add_argument("--nms_threshold",
                        default=None,
                        type=float)  # 指定非最大抑制阈值
    parser.add_argument("--stage_one_K", type=int)  # 第一阶段的k值
    parser.add_argument("--master_port", default='12355')  # 主端口
    parser.add_argument("--gpu_split",
                        nargs='?',
                        default=0,
                        const=2,
                        type=int)  # GPU分割设置
    parser.add_argument("--waymo",
                        action='store_true')  # waymo数据集
    parser.add_argument("--argoverse",
                        action='store_true')  # argoverse数据集
    parser.add_argument("--argoverse2",
                        action='store_true')  # argoverse2数据集
    parser.add_argument("--nuscenes",
                        action='store_true')  # nuscenes数据集
    parser.add_argument("--future_frame_num",
                        default=60,
                        type=int)  # 指定未来训练帧的帧数
    parser.add_argument("--future_test_frame_num",
                        default=16,
                        type=int)  # 制定未来测试帧的数量
    parser.add_argument("--single_agent",
                        action='store_true',
                        default=True)  # 是否使用单个代理
    parser.add_argument("--agent_type",
                        default=None,
                        type=str)  # 代理类型
    parser.add_argument("--inter_agent_types",
                        default=None,
                        nargs=2,
                        type=str)  # 交互代理类型
    parser.add_argument("--mode_num",
                        default=6,
                        type=int)  # 模式数量

class Args:
    data_dir = None
    data_kind = None
    debug = None
    train_batch_size = None
    seed = None
    eval_batch_size = None
    distributed_training = None
    cuda_visible_device_num = None
    log_dir = None
    learning_rate = None
    do_eval = None
    hidden_size = None
    sub_graph_depth = None
    global_graph_depth = None
    train_batch_size = None
    num_train_epochs = None
    initializer_range = None
    sub_graph_batch_size = None
    temp_file_dir = None
    output_dir = None
    use_map = None
    reuse_temp_file = None
    old_version = None
    model_recover_path = None
    do_train = None
    max_distance = None
    no_sub_graph = None
    other_params: Dict = None
    eval_params = None
    train_params = None
    no_agents = None
    not_use_api = None
    core_num = None
    visualize = None
    train_extra = None
    hidden_dropout_prob = None
    use_centerline = None
    autoregression = None
    lstm = None
    add_prefix = None
    attention_decay = None
    do_test = None
    placeholder = None
    multi = None
    method_span = None
    waymo = None
    argoverse = None
    argoverse2 = None
    nuscenes = None
    single_agent = None
    agent_type = None
    future_frame_num = None
    no_cuda = None
    mode_num = None
    nms_threshold = None
    inter_agent_types = None

def parse_args():
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args: Args = parser.parse_args()
    return args


def load_pre_trained_model(model_path, args, device=None):
    model = VectorNet(args)
    model_recover = torch.load(model_path)
    print(model_recover)
    model.load_state_dict(model_recover)


if __name__ == '__main__':
    args = parse_args()
    model_recover_path = args.model_recover_path
    result = load_pre_trained_model(model_recover_path, args)
    print('result')
    print(result)
