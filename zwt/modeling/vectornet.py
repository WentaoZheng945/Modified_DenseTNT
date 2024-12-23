from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from modeling.decoder import Decoder, DecoderResCat
from modeling.lib import MLP, GlobalGraph, LayerNorm, CrossAttention, GlobalGraphRes
import utils


class NewSubGraph(nn.Module):
    """
    这个类首先使用2层MLP, 然后使用三层GlobalGraph（transformer的encoder）
    所以此模型的输入为(多个交通参与者，时间，128)，这里判断的是每个交通参与者不同时间之间的关系，我理解这个地方也符合轨迹预测的基本盘
    """

    def __init__(self, hidden_size, depth=None):
        super(NewSubGraph, self).__init__()
        if depth is None:
            depth = args.sub_graph_depth  # 3
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])

        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_3 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_4 = nn.ModuleList([GlobalGraph(hidden_size) for _ in range(depth)])
        self.layer_0_again = MLP(hidden_size)

    def forward(self, input_list: list):
        batch_size = len(input_list)
        device = input_list[0].device
        hidden_states, lengths = utils.merge_tensors(input_list, device)  # 把List:[torch.Tensor]拼成一个tensor,返回：Tuple[Tensor, List[int]]
        hidden_size = hidden_states.shape[2]  # 解析场景后隐藏层的大小
        max_vector_num = hidden_states.shape[1]  # 最大的向量长度

        attention_mask = torch.zeros([batch_size, max_vector_num, max_vector_num], device=device)
        hidden_states = self.layer_0(hidden_states)
        hidden_states = self.layer_0_again(hidden_states)  # 输出的shape为（batch_size, max_vector_num, 128）
        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            # hidden_states = layer(hidden_states, attention_mask)
            # hidden_states = self.layers_2[layer_index](hidden_states)
            # hidden_states = F.relu(hidden_states) + temp
            # self-Attention
            hidden_states = layer(hidden_states, attention_mask)  # (batch_size, max_vector_num, 128)
            # 激活层
            hidden_states = F.relu(hidden_states)
            # 残差
            hidden_states = hidden_states + temp
            # Normalize
            hidden_states = self.layers_2[layer_index](hidden_states)  # (batch_size, max_vector_num, 128)

        # （batch_size, 128）和 List[(length, 128)]
        return torch.max(hidden_states, dim=1)[0], torch.cat(utils.de_merge_tensors(hidden_states, lengths))


class VectorNet(nn.Module):
    r"""
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """

    def __init__(self, args_: utils.Args):
        super(VectorNet, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size  # 128

        self.point_level_sub_graph = NewSubGraph(hidden_size)
        self.point_level_cross_attention = CrossAttention(hidden_size)

        self.global_graph = GlobalGraph(hidden_size)

        # Use multi-head attention and residual connection.
        if 'enhance_global_graph' in args.other_params:
            self.global_graph = GlobalGraphRes(hidden_size)
        if 'laneGCN' in args.other_params:
            self.laneGCN_A2L = CrossAttention(hidden_size)
            self.laneGCN_L2L = GlobalGraphRes(hidden_size)
            self.laneGCN_L2A = CrossAttention(hidden_size)

        self.decoder = Decoder(args, self)

        if 'complete_traj' in args.other_params:
            self.decoder.complete_traj_cross_attention = CrossAttention(hidden_size)
            self.decoder.complete_traj_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=self.decoder.future_frame_num * 2)

    def forward_encode_sub_graph(self, mapping: List[Dict], matrix: List[np.ndarray], polyline_spans: List[List[slice]],
                                 device, batch_size) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
        input_list_list = []
        # TODO(cyrushx): This is not used? Is it because input_list_list includes map data as well?
        # Yes, input_list_list includes map data, this will be used in the future release.
        map_input_list_list = []
        lane_states_batch = None
        for i in range(batch_size):
            input_list = []
            map_input_list = []
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(matrix[i][polyline_span], device=device)
                input_list.append(tensor)  # 动态参与者（所有时刻）+ 车道信息（车道中所有点）
                if j >= map_start_polyline_idx:
                    map_input_list.append(tensor)  # 地图信息

            input_list_list.append(input_list)  # 整个场景的交通参与者信息
            map_input_list_list.append(map_input_list)  # 整个场景的地图信息

        if True:
            element_states_batch = []
            for i in range(batch_size):
                a, b = self.point_level_sub_graph(input_list_list[i])
                element_states_batch.append(a)  # List[(vehicle_nums, 128)]

        if 'lane_scoring' in args.other_params:
            lane_states_batch = []
            for i in range(batch_size):
                a, b = self.point_level_sub_graph(map_input_list_list[i])
                lane_states_batch.append(a)

        # We follow laneGCN to fuse realtime traffic information from agent nodes to lane nodes.
        if 'laneGCN' in args.other_params:
            for i in range(batch_size):
                map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
                agents = element_states_batch[i][:map_start_polyline_idx]
                lanes = element_states_batch[i][map_start_polyline_idx:]
                # Origin laneGCN contains three fusion layers. Here one fusion layer is enough.
                if True:
                    # 这里融合了车道信息以及（车道信息+预测车辆的信息）的信息
                    lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents[0:1]]).unsqueeze(0)).squeeze(0)
                    # lanes的shape为(lane_nums, 128)
                    # lane_GCN的输入为（1, lane_nums, 128)
                    # 输出为(lane_nums, 128)
                else:
                    lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), agents.unsqueeze(0)).squeeze(0)
                    lanes = lanes + self.laneGCN_L2L(lanes.unsqueeze(0)).squeeze(0)
                    agents = agents + self.laneGCN_L2A(agents.unsqueeze(0), lanes.unsqueeze(0)).squeeze(0)
                element_states_batch[i] = torch.cat([agents, lanes])  # 拼接车辆信息以及预测车辆与地图融合后的信息
        # return List[(lane_nums+car_nums, 128)], List[(lane_nums, 128)]
        return element_states_batch, lane_states_batch

    # @profile
    def forward(self, mapping: List[Dict], device):
        import time
        global starttime
        starttime = time.time()

        matrix = utils.get_from_mapping(mapping, 'matrix')
        # TODO(cyrushx): Can you explain the structure of polyline spans?
        # vectors of i_th element is matrix[polyline_spans[i]]
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')

        batch_size = len(matrix)
        # for i in range(batch_size):
        # polyline_spans[i] = [slice(polyline_span[0], polyline_span[1]) for polyline_span in polyline_spans[i]]

        if args.argoverse:
            utils.batch_init(mapping)

        element_states_batch, lane_states_batch = self.forward_encode_sub_graph(mapping, matrix, polyline_spans, device, batch_size)  # List[(lane_nums+car_nums, 128)], List[(lane_nums, 128)]

        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)  # (batch_size, max(lane_nums+car_nums), 128) List[lanes_nums+car_nums, ...]
        max_poly_num = max(inputs_lengths)  # max(lane_nums+car_nums)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask, mapping)  # # (batch, max(lane_nums+car_nums), 128)

        utils.logging('time3', round(time.time() - starttime, 2), 'secs')

        return self.decoder(mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device)

"""
VectorNet是一个分级图神经网络：
首先用polyline subgraphs从vector中抽取实例的特征，比如每辆车和行人的轨迹特征，每个车道lane的特征，每个人行道的特征。
在第二阶段，global interaction graph是一个基于attention的全连接图，可以让不同实例进行信息交换，比如建模车和车之间的交互关系，车和车道之间的关系，车道和车道之间的关系等等。
当行人走上斑马线时，代表斑马线的vector和代表行人的vector之间就会产生信息交换引导模型关注他们之间的联系。
"""
