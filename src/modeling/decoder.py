from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import structs
import utils_cython
from modeling.lib import PointSubGraph, GlobalGraphRes, CrossAttention, GlobalGraph, MLP

import utils


class DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)  # (batch_size, max_vector, 128) + (batch_size, max_vector, 128)
        hidden_states = self.fc(hidden_states)  # (batch_size, max_vector, 2)
        return hidden_states


class DecoderResCat(nn.Module):
    """解码，得到得分"""
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class Decoder(nn.Module):

    def __init__(self, args_: utils.Args, vectornet):
        super(Decoder, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size  # 128
        self.future_frame_num = args.future_frame_num  # 60
        self.mode_num = args.mode_num  # 6

        self.decoder = DecoderRes(hidden_size, out_features=2)  # (batch_size, max_vector, 2)

        if 'variety_loss' in args.other_params:
            self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2)

            if 'variety_loss-prob' in args.other_params:
                self.variety_loss_decoder = DecoderResCat(hidden_size, hidden_size, out_features=6 * self.future_frame_num * 2 + 6)
        elif 'goals_2D' in args.other_params:
            # self.decoder = DecoderResCat(hidden_size, hidden_size, out_features=self.future_frame_num * 2)
            self.goals_2D_mlps = nn.Sequential(
                MLP(2, hidden_size),
                MLP(hidden_size),
                MLP(hidden_size)
            )  # 输入：(batch_size, max_vector, 2) 输出：(batch_size, max_vector, 128)
            # self.goals_2D_decoder = DecoderRes(hidden_size * 3, out_features=1)
            self.goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)  # 输入为(batch_size, max_vector, in_features) 输出为(batch_size, max_vector, out_features)
            self.goals_2D_cross_attention = CrossAttention(hidden_size)  # 输出为(batch, max_vector_num, hidden_size)
            # Fuse goal feature and agent feature when encoding goals.
            if 'point_sub_graph' in args.other_params:
                self.goals_2D_point_sub_graph = PointSubGraph(hidden_size)  # 输入：# (1, 'goal num', 2)  (1, hidden_size) 输出为(1, goal_num, 128)

        if 'lane_scoring' in args.other_params:
            self.stage_one_cross_attention = CrossAttention(hidden_size)  # 输出为(1, lane_num, 128)
            self.stage_one_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=1)
            self.stage_one_goals_2D_decoder = DecoderResCat(hidden_size, hidden_size * 4, out_features=1)

        if 'set_predict' in args.other_params:  # 训练第二阶段时，只会更新与set_predict有关的参数
            if args.do_train:
                if 'set_predict-train_recover' in args.other_params:
                    model_recover = torch.load(args.other_params['set_predict-train_recover'])
                else:
                    model_recover = torch.load(args.model_recover_path)
                vectornet.decoder = self
                utils.load_model(vectornet, model_recover)  # 把参数赋给Vectornet
                # self must be vectornet
                for p in vectornet.parameters():
                    p.requires_grad = False  # 训练goal set prediction时不会更新vectornet参数

            self.set_predict_point_feature = nn.Sequential(MLP(3, hidden_size), MLP(hidden_size, hidden_size))

            self.set_predict_encoders = nn.ModuleList(
                [GlobalGraphRes(hidden_size) for _ in range(args.other_params['set_predict'])])

            self.set_predict_decoders = nn.ModuleList(
                [DecoderResCat(hidden_size, hidden_size * 2, out_features=13) for _ in range(args.other_params['set_predict'])])

    def lane_scoring(self, i, mapping, lane_states_batch, inputs, inputs_lengths,
                     hidden_states, device, loss):
        """
        选出概率最大的k条车道(概率超过95)，返回其车道编码
        Args:
            i: example index in batch
            mapping:
            lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
            inputs:hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
            inputs_lengths: valid element number of each example
            hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, 'element num', hidden_size])
            device: GPU
            loss: (shape [batch_size])

        Returns: (shape [selected_lanes, 128])

        """
        def get_stage_one_scores():
            stage_one_hidden = lane_states_batch[i]  # shape (lane num, hidden_size)
            # 输入为(1, lane_num, 128), (1, element_num, 128) 输出为(lane_num, 128)
            stage_one_hidden_attention = self.stage_one_cross_attention(
                stage_one_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)
            # 输入为(lane_num, 128) 输出为(lane_num, 1)
            stage_one_scores = self.stage_one_decoder(torch.cat([hidden_states[i, 0, :].unsqueeze(0).expand(
                stage_one_hidden.shape), stage_one_hidden, stage_one_hidden_attention], dim=-1))
            stage_one_scores = stage_one_scores.squeeze(-1)  # (lane_num,)
            stage_one_scores = F.log_softmax(stage_one_scores, dim=-1)  # 转换概率，输出也为(lane_num, )
            return stage_one_scores  # (lane_num, )

        stage_one_scores = get_stage_one_scores()  # (lane_num, )
        if args.argoverse2:
            assert len(stage_one_scores) == len(mapping[i]['polygons']) // 2
        else:
            assert len(stage_one_scores) == len(mapping[i]['polygons'])
        # 输出一个值，这个值表示输出的概率分布与真实标签之间的差异
        loss[i] += F.nll_loss(stage_one_scores.unsqueeze(0),
                              torch.tensor([mapping[i]['stage_one_label']], device=device))

        # Select top K lanes, where K is dynamic. The sum of the probabilities of selected lanes is larger than threshold (0.95).
        if True:
            # 返回值第一项为stage_one_scores中最大的k个元素(按大小顺序排列)，第二项为k个元素对应的索引
            _, stage_one_topk_ids = torch.topk(stage_one_scores, k=len(stage_one_scores))
            threshold = 0.95
            sum = 0.0
            for idx, each in enumerate(torch.exp(stage_one_scores[stage_one_topk_ids])):
                sum += each
                if sum > threshold:
                    stage_one_topk_ids = stage_one_topk_ids[:idx + 1]
                    break
            utils.other_errors_put('stage_one_k', len(stage_one_topk_ids))
        # _, stage_one_topk_ids = torch.topk(stage_one_scores, k=min(args.stage_one_K, len(stage_one_scores)))

        if mapping[i]['stage_one_label'] in stage_one_topk_ids.tolist():
            # 终点车道在选出的车道内
            utils.other_errors_put('stage_one_recall', 1.0)
        else:
            # 终点车道不在选出的车道内
            utils.other_errors_put('stage_one_recall', 0.0)

        topk_lanes = lane_states_batch[i][stage_one_topk_ids]
        return topk_lanes  # 返回所选出的车道的编码 shape为(selected_lanes, 128)

    def get_scores_of_dense_goals(self, i, goals_2D, mapping, labels, device, scores,
                                  get_scores_inputs, gt_points=None):
        if args.argoverse:
            k = 150
        else:
            k = 40
        _, topk_ids = torch.topk(scores, k=min(k, len(scores)))
        topk_ids = topk_ids.tolist()  # 分数前k高goal的索引

        # Sample dense goals from top K sparse goals.（默认情况下，一个点变25个点），List[(x, y), (x, y)]
        goals_2D_dense = utils.get_neighbour_points(goals_2D[topk_ids], topk_ids=topk_ids, mapping=mapping[i])

        # shape(dense_goal_nums, 2) shape(goal_nums, 2)
        goals_2D_dense = torch.cat([torch.tensor(goals_2D_dense, device=device, dtype=torch.float),
                                    torch.tensor(goals_2D, device=device, dtype=torch.float)], dim=0)

        old_vector_num = len(goals_2D)

        scores = self.get_scores(goals_2D_dense, *get_scores_inputs)  # 返回值为log scores of goals (shape ['goal num'])

        index = torch.argmax(scores).item()
        point = np.array(goals_2D_dense[index].tolist())  # 终点坐标

        if not args.do_test:
            label = np.array(labels[i]).reshape([self.future_frame_num, 2])
            final_idx = mapping[i].get('final_idx', -1)
            mapping[i]['goals_2D_labels'] = np.argmin(utils.get_dis(goals_2D_dense.detach().cpu().numpy(), label[final_idx]))

        return scores, point, goals_2D_dense.detach().cpu().numpy()

    def goals_2D_per_example_calc_loss(self, i: int, goals_2D: np.ndarray, mapping: List[Dict], inputs: Tensor,
                                       inputs_lengths: List[int], hidden_states: Tensor, device, loss: Tensor,
                                       DE: np.ndarray, gt_points: np.ndarray, scores: Tensor, highest_goal: np.ndarray,
                                       labels_is_valid: List[np.ndarray]):
        """
        Calculate loss for a training example
        Args:
            i: example index in batch
            goals_2D: candidate dense goals (shape ['dense goal num', 2])
            mapping:
            inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
            inputs_lengths: valid element number of each example
            hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
            device: GPU
            loss: (shape [batch_size])
            DE: displacement error (shape [batch_size, self.future_frame_num])
            gt_points: 未来帧的真实轨迹点 shape(60, 2)
            scores: shape(goal_nums,)
            highest_goal:shape([2, ])
            labels_is_valid: List[(60, )]

        Returns:

        """
        final_idx = mapping[i].get('final_idx', -1)
        gt_goal = gt_points[final_idx]
        DE[i][final_idx] = np.sqrt((highest_goal[0] - gt_points[final_idx][0]) ** 2 + (highest_goal[1] - gt_points[final_idx][1]) ** 2)  # 采集得到的终点坐标与实际终点坐标的误差
        if 'complete_traj' in args.other_params:
            target_feature = self.goals_2D_mlps(torch.tensor(gt_points[final_idx], dtype=torch.float, device=device))  # 输出为(128)
            pass
            if True:
                target_feature.detach_()
                # complete_traj_cross_attention和complete_traj_decoder共同完成了基于目标点位的轨迹预测
                # 输入为(1, 1, 128) (1, elements_num, 128) 输出为 (128, )
                hidden_attention = self.complete_traj_cross_attention(
                    target_feature.unsqueeze(0).unsqueeze(0), inputs[i][:inputs_lengths[i]].detach().unsqueeze(0)).squeeze(
                    0).squeeze(0)
                predict_traj = self.complete_traj_decoder(
                    torch.cat([hidden_states[i, 0, :].detach(), target_feature, hidden_attention], dim=-1)).view(
                    [self.future_frame_num, 2])  # 输出为shape(60, 2)
            loss[i] += (F.smooth_l1_loss(predict_traj, torch.tensor(gt_points, dtype=torch.float, device=device), reduction='none') * \
                        torch.tensor(labels_is_valid[i], dtype=torch.float, device=device).view(self.future_frame_num, 1)).mean()

        # nll_loss求的是负对数似然损失：即模型预测的类别概率分布与真实类别标签之间的差异。
        loss[i] += F.nll_loss(scores.unsqueeze(0),
                              torch.tensor([mapping[i]['goals_2D_labels']], device=device))

    def goals_2D_per_example(self, i: int, goals_2D: np.ndarray, mapping: List[Dict], lane_states_batch: List[Tensor],
                             inputs: Tensor, inputs_lengths: List[int], hidden_states: Tensor, labels: List[np.ndarray],
                             labels_is_valid: List[np.ndarray], device, loss: Tensor, DE: np.ndarray):
        """
        :param i: example index in batch
        :param goals_2D: candidate goals sampled from map (shape ['goal num', 2])
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param loss: (shape [batch_size])
        :param DE: displacement error (shape [batch_size, self.future_frame_num])
        """
        if args.do_train:
            final_idx = mapping[i].get('final_idx', -1)
            assert labels_is_valid[i][final_idx]

        gt_points = labels[i].reshape([self.future_frame_num, 2])

        topk_lanes = None
        # Get top K lanes with the highest probability.
        if 'lane_scoring' in args.other_params:
            topk_lanes = self.lane_scoring(i, mapping, lane_states_batch, inputs, inputs_lengths,
                                           hidden_states, device, loss)

        get_scores_inputs = (inputs, hidden_states, inputs_lengths, i, mapping, device, topk_lanes)

        # There is a lane scoring module (see Section 3.2) in the paper in order to reduce the number of goal candidates.
        # In this implementation, we use goal scoring instead of lane scoring, because we observed that it performs slightly better than lane scoring.
        # Here goals_2D are sparse cnadidate goals sampled from map.
        if 'goal_scoring' in args.other_params:
            goals_2D_tensor = torch.tensor(goals_2D, device=device, dtype=torch.float)
            scores = self.get_scores(goals_2D_tensor, *get_scores_inputs)  # 由高可能性车道得到所有可行终点目标得分
            index = torch.argmax(scores).item()  # int
            highest_goal = goals_2D[index]

        # Get dense goals and their scores.
        # With the help of the above goal scoring, we can reduce the number of dense goals.
        # After this step, goals_2D become dense goals.
        # scores:shape([goals_num, ]) highest_goal:shape([2, ]) goals_2D:ndarray shape(goals_num, 2)
        # 这一步就是基于高概率的点生成dense goals
        scores, highest_goal, goals_2D = \
            self.get_scores_of_dense_goals(i, goals_2D, mapping, labels, device, scores,
                                           get_scores_inputs, gt_points)

        if args.do_train:
            self.goals_2D_per_example_calc_loss(i, goals_2D, mapping, inputs, inputs_lengths,
                                                hidden_states, device, loss, DE, gt_points, scores, highest_goal, labels_is_valid)

        if args.visualize:
            mapping[i]['vis.goals_2D'] = goals_2D
            mapping[i]['vis.scores'] = np.array(scores.tolist())
            mapping[i]['vis.labels'] = gt_points
            mapping[i]['vis.labels_is_valid'] = labels_is_valid[i]
        if args.visualize_post:
            mapping[i]['vis_post.goals_2D'] = goals_2D
            mapping[i]['vis_post.scores'] = np.array(scores.tolist())
            mapping[i]['vis_post.labels'] = gt_points
            mapping[i]['vis_post.labels_is_valid'] = labels_is_valid[i]

        if 'set_predict' in args.other_params:
            self.run_set_predict(goals_2D, scores, mapping, device, loss, i)
            if args.visualize:
                set_predict_ans_points = mapping[i]['set_predict_ans_points']
                predict_trajs = np.zeros((6, self.future_frame_num, 2))
                predict_trajs[:, -1, :] = set_predict_ans_points

        else:
            if args.do_eval:
                if args.nms_threshold is not None:
                    utils.select_goals_by_NMS(mapping[i], goals_2D, np.array(scores.tolist()), args.nms_threshold, mapping[i]['speed'])
                elif 'optimization' in args.other_params:
                    mapping[i]['goals_2D_scores'] = goals_2D.astype(np.float32), np.array(scores.tolist(), dtype=np.float32)
                else:
                    assert False

    def goals_2D_eval(self, batch_size, mapping, labels, hidden_states, inputs, inputs_lengths, device):
        if 'set_predict' in args.other_params:
            pred_goals_batch = [mapping[i]['set_predict_ans_points'] for i in range(batch_size)]
            pred_probs_batch = np.zeros((batch_size, 6))
        elif 'optimization' in args.other_params:
            pred_goals_batch, pred_probs_batch = utils.select_goals_by_optimization(
                np.array(labels).reshape([batch_size, self.future_frame_num, 2]), mapping)  # 通过优化的方法得到了这一批goals中最好的6个点
        elif args.nms_threshold is not None:
            pred_goals_batch = [mapping[i]['pred_goals'] for i in range(batch_size)]
            pred_probs_batch = [mapping[i]['pred_probs'] for i in range(batch_size)]
        else:
            assert False

        pred_goals_batch = np.array(pred_goals_batch)
        pred_probs_batch = np.array(pred_probs_batch)
        assert pred_goals_batch.shape == (batch_size, self.mode_num, 2)
        assert pred_probs_batch.shape == (batch_size, self.mode_num)

        if 'complete_traj' in args.other_params:
            pred_trajs_batch = []
            for i in range(batch_size):
                targets_feature = self.goals_2D_mlps(torch.tensor(pred_goals_batch[i], dtype=torch.float, device=device))
                hidden_attention = self.complete_traj_cross_attention(
                    targets_feature.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)
                predict_trajs = self.complete_traj_decoder(
                    torch.cat([hidden_states[i, 0, :].unsqueeze(0).expand(len(targets_feature), -1), targets_feature,
                               hidden_attention], dim=-1)).view([self.mode_num, self.future_frame_num, 2])  # 输出shape为(6, 60, 2)
                predict_trajs = np.array(predict_trajs.tolist())
                final_idx = mapping[i].get('final_idx', -1)
                predict_trajs[:, final_idx, :] = pred_goals_batch[i]  # 最后一帧换成终点
                mapping[i]['vis.predict_trajs'] = predict_trajs.copy()
                if args.visualize_post:
                    mapping[i]['vis_post.predict_trajs'] = predict_trajs.copy()

                if args.argoverse:
                    for each in predict_trajs:
                        utils.to_origin_coordinate(each, i)
                pred_trajs_batch.append(predict_trajs)
            pred_trajs_batch = np.array(pred_trajs_batch)
        else:
            pass
        if args.visualize:
            for i in range(batch_size):
                utils.visualize_goals_2D(mapping[i], mapping[i]['vis.goals_2D'], mapping[i]['vis.scores'], self.future_frame_num,
                                         labels=mapping[i]['vis.labels'],
                                         labels_is_valid=mapping[i]['vis.labels_is_valid'],
                                         predict=mapping[i]['vis.predict_trajs'],
                                         extra=True)

        return pred_trajs_batch, pred_probs_batch, None

    def variety_loss(self, mapping: List[Dict], hidden_states: Tensor, batch_size, inputs: Tensor,
                     inputs_lengths: List[int], labels_is_valid: List[np.ndarray], loss: Tensor,
                     DE: np.ndarray, device, labels: List[np.ndarray]):
        """
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param DE: displacement error (shape [batch_size, self.future_frame_num])
        """
        outputs = self.variety_loss_decoder(hidden_states[:, 0, :])
        pred_probs = None
        if 'variety_loss-prob' in args.other_params:
            pred_probs = F.log_softmax(outputs[:, -6:], dim=-1)
            outputs = outputs[:, :-6].view([batch_size, 6, self.future_frame_num, 2])
        else:
            outputs = outputs.view([batch_size, 6, self.future_frame_num, 2])

        for i in range(batch_size):
            if args.do_train:
                assert labels_is_valid[i][-1]
            gt_points = np.array(labels[i]).reshape([self.future_frame_num, 2])
            argmin = np.argmin(utils.get_dis_point_2_points(gt_points[-1], np.array(outputs[i, :, -1, :].tolist())))

            loss_ = F.smooth_l1_loss(outputs[i, argmin],
                                     torch.tensor(gt_points, device=device, dtype=torch.float), reduction='none')
            loss_ = loss_ * torch.tensor(labels_is_valid[i], device=device, dtype=torch.float).view(self.future_frame_num, 1)
            if labels_is_valid[i].sum() > utils.eps:
                loss[i] += loss_.sum() / labels_is_valid[i].sum()

            if 'variety_loss-prob' in args.other_params:
                loss[i] += F.nll_loss(pred_probs[i].unsqueeze(0), torch.tensor([argmin], device=device))
        if args.do_eval:
            outputs = np.array(outputs.tolist())
            pred_probs = np.array(pred_probs.tolist(), dtype=np.float32) if pred_probs is not None else pred_probs
            for i in range(batch_size):
                for each in outputs[i]:
                    utils.to_origin_coordinate(each, i)

            return outputs, pred_probs, None
        return loss.mean(), DE, None

    def forward(self, mapping: List[Dict], batch_size, lane_states_batch: List[Tensor], inputs: Tensor,
                inputs_lengths: List[int], hidden_states: Tensor, device):
        """
        :param lane_states_batch: each value in list is hidden states of lanes (value shape ['lane num', hidden_size])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example （List[lanes_nums+car_nums, ...]）
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, 'element num', hidden_size])
        """
        labels = utils.get_from_mapping(mapping, 'labels')  # List[ndarray] 目标车未来所有帧坐标的真值(原笛卡尔坐标系下)
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')  # list[ndarray]都为1
        loss = torch.zeros(batch_size, device=device)
        DE = np.zeros([batch_size, self.future_frame_num])  # 位移误差

        if 'variety_loss' in args.other_params:
            return self.variety_loss(mapping, hidden_states, batch_size, inputs, inputs_lengths, labels_is_valid, loss, DE, device, labels)
        elif 'goals_2D' in args.other_params:
            for i in range(batch_size):
                goals_2D = mapping[i]['goals_2D']  # List[(nums,2)]

                self.goals_2D_per_example(i, goals_2D, mapping, lane_states_batch, inputs, inputs_lengths,
                                          hidden_states, labels, labels_is_valid, device, loss, DE)

            if 'set_predict' in args.other_params:
                pass
                # if args.do_eval:
                #     pred_trajs_batch = np.zeros([batch_size, 6, self.future_frame_num, 2])
                #     for i in range(batch_size):
                #         if 'set_predict_trajs' in mapping[i]:
                #             pred_trajs_batch[i] = mapping[i]['set_predict_trajs']
                #             for each in pred_trajs_batch[i]:
                #                 utils.to_origin_coordinate(each, i)
                #     return pred_trajs_batch

            if args.do_eval:
                return self.goals_2D_eval(batch_size, mapping, labels, hidden_states, inputs, inputs_lengths, device)
            else:
                if args.visualize:
                    for i in range(batch_size):
                        predict = np.zeros((self.mode_num, self.future_frame_num, 2))
                        utils.visualize_goals_2D(mapping[i], mapping[i]['vis.goals_2D'], mapping[i]['vis.scores'],
                                                 self.future_frame_num,
                                                 labels=mapping[i]['vis.labels'],
                                                 labels_is_valid=mapping[i]['vis.labels_is_valid'],
                                                 predict=predict)
                return loss.mean(), DE, None
        else:
            assert False

    def get_scores(self, goals_2D_tensor: Tensor, inputs, hidden_states, inputs_lengths, i, mapping, device, topk_lanes):
        """
        :param goals_2D_tensor: candidate goals sampled from map (shape ['goal num', 2])
        :param inputs: hidden states of all elements before encoding by global graph (shape [batch_size, 'element num', hidden_size])
        :param inputs_lengths: valid element number of each example
        :param hidden_states: hidden states of all elements after encoding by global graph (shape [batch_size, -1, hidden_size])
        :param topk_lanes: hidden states of topk lanes (shape ['selected lanes', 128])
        :return: log scores of goals (shape ['goal num'])
        """
        # Fuse goal feature and agent feature when encoding goals.
        if 'point_sub_graph' in args.other_params:
            # 输出的shape为(goal_num, 128)
            goals_2D_hidden = self.goals_2D_point_sub_graph(goals_2D_tensor.unsqueeze(0), hidden_states[i, 0:1, :]).squeeze(0)  # (1, 'goal num', 2)  (1, hidden_size)
        else:
            goals_2D_hidden = self.goals_2D_mlps(goals_2D_tensor)

        # 输入shape为(1, goal_num, 128) (1, element_num, 128)
        # 输出shape为(goal_num, 128)
        goals_2D_hidden_attention = self.goals_2D_cross_attention(
            goals_2D_hidden.unsqueeze(0), inputs[i][:inputs_lengths[i]].unsqueeze(0)).squeeze(0)

        if 'lane_scoring' in args.other_params:
            # Perform cross attention from goals to top K lanes. It's a trick to improve model performance.
            # 输入shape为(1, goal_num, 128) (1, selected_num, 128)
            # 输出shape为(goal_num, 128)
            stage_one_goals_2D_hidden_attention = self.goals_2D_cross_attention(
                goals_2D_hidden.unsqueeze(0), topk_lanes.unsqueeze(0)).squeeze(0)
            # 输入shape为(1, 128),expand后为(goal_num, 128)
            # li = [shape(goal_num, 128), shape(goal_num, 128), shape(goal_num, 128), shape(goal_num, 128)]
            li = [hidden_states[i, 0, :].unsqueeze(0).expand(goals_2D_hidden.shape),
                  goals_2D_hidden, goals_2D_hidden_attention, stage_one_goals_2D_hidden_attention]

            scores = self.stage_one_goals_2D_decoder(torch.cat(li, dim=-1))  # 输入为(goal_num, 128*4) 输出为(goal_num, 1)
        else:
            scores = self.goals_2D_decoder(torch.cat([hidden_states[i, 0, :].unsqueeze(0).expand(
                goals_2D_hidden.shape), goals_2D_hidden, goals_2D_hidden_attention], dim=-1))

        scores = scores.squeeze(-1)  # 输出为(goal_num, )
        scores = F.log_softmax(scores, dim=-1)
        return scores  # 返回值为log scores of goals (shape ['goal num'])

    def run_set_predict(self, goals_2D, scores, mapping, device, loss, i):
        """

        Args:
            goals_2D: 是dense goals
            scores:  dense goals对应的分数
            mapping:
            device:
            loss: shape(batch_size, ) 此函数需做置空处理
            i:

        Returns:

        """
        gt_points = mapping[i]['labels'].reshape((self.future_frame_num, 2))

        if args.argoverse:
            if 'set_predict-topk' in args.other_params:
                topk_num = args.other_params['set_predict-topk']

                if topk_num == 0:
                    topk_num = torch.sum(scores > np.log(0.00001)).item()  # 概率大于0.00001的目标点的数量

                _, topk_ids = torch.topk(scores, k=min(topk_num, len(scores)))
                goals_2D = goals_2D[topk_ids.cpu().numpy()]  # 选出的大概率的终点
                scores = scores[topk_ids]  # 对应上面点的得分

        scores_positive_np = np.exp(np.array(scores.tolist(), dtype=np.float32))  # 将分数转换为正数
        goals_2D = goals_2D.astype(np.float32)

        max_point_idx = torch.argmax(scores)  # 得分最高的点的索引
        vectors_3D = torch.cat([torch.tensor(goals_2D, device=device, dtype=torch.float), scores.unsqueeze(1)], dim=-1)  # shape([goal num, 3])
        vectors_3D = torch.tensor(vectors_3D.tolist(), device=device, dtype=torch.float)

        vectors_3D[:, 0] -= goals_2D[max_point_idx, 0]  # 把vector_3D的第一列变成相对于最高得分的相对坐标(x)
        vectors_3D[:, 1] -= goals_2D[max_point_idx, 1]  # 把vector_3D的第二列变成相对于最高得分的相对坐标(y)

        points_feature = self.set_predict_point_feature(vectors_3D)  # 输入：shape(goals, 3) 输出：shape(goals, 128)
        costs = np.zeros(args.other_params['set_predict'])  # shape(6, )  # 储存每一层解码器输出的预测目标点集合与真实目标点集合的匹配程度
        pseudo_labels = []  # 全是None
        predicts = []  # 每一层解码器预测的目标轨迹点

        set_predict_trajs_list = []
        group_scores = torch.zeros([len(self.set_predict_encoders)], device=device)  # shape(6, )  # 每一层解码器直接输出的分数

        start_time = utils.time.time()

        if True:
            for k, (encoder, decoder) in enumerate(zip(self.set_predict_encoders, self.set_predict_decoders)):
                if 'set_predict-one_encoder' in args.other_params:
                    encoder = self.set_predict_encoders[0]

                if True:
                    if 'set_predict-one_encoder' in args.other_params and k > 0:
                        pass
                    else:
                        # 输入为(1, goals, 128) 输出为(goals, 128)
                        encoding = encoder(points_feature.unsqueeze(0)).squeeze(0)

                    # (128, ) 和(128, )合并为(128*2, ) 输出为(13, )
                    decoding = decoder(torch.cat([torch.max(encoding, dim=0)[0], torch.mean(encoding, dim=0)], dim=-1)).view([13])
                    group_scores[k] = decoding[0]  # decoder的第一个元素是分数
                    predict = decoding[1:].view([6, 2])  # 后面12个是预测的轨迹

                    predict[:, 0] += goals_2D[max_point_idx, 0]  # 还原为真实坐标
                    predict[:, 1] += goals_2D[max_point_idx, 1]

                predicts.append(predict)

                if args.do_eval:
                    pass
                else:
                    selected_points = np.array(predict.tolist(), dtype=np.float32)
                    temp = None
                    assert goals_2D.dtype == np.float32, goals_2D.dtype
                    kwargs = None
                    if 'set_predict-MRratio' in args.other_params:
                        kwargs = {}
                        kwargs['set_predict-MRratio'] = args.other_params['set_predict-MRratio']  # 默认为0.0
                    costs[k] = utils_cython.set_predict_get_value(goals_2D, scores_positive_np, selected_points, kwargs=kwargs)

                    pseudo_labels.append(temp)  # 初始最优的y^

        argmin = torch.argmax(group_scores).item()  # 最小分数索引

        if args.do_train:
            utils.other_errors_put('set_hungary', np.min(costs))
            group_scores = F.log_softmax(group_scores, dim=-1)
            min_cost_idx = np.argmin(costs)
            loss[i] = 0

            if True:
                selected_points = np.array(predicts[min_cost_idx].tolist(), dtype=np.float32)  # 6层解码器输出的最好的目标点
                kwargs = None
                if 'set_predict-MRratio' in args.other_params:
                    kwargs = {}
                    kwargs['set_predict-MRratio'] = args.other_params['set_predict-MRratio']
                _, dynamic_label = utils_cython.set_predict_next_step(goals_2D, scores_positive_np, selected_points,
                                                                      lr=args.set_predict_lr, kwargs=kwargs)  # 不断扰动找到更好的点
                # loss[i] += 2.0 / globals.set_predict_lr * \
                #            F.l1_loss(predicts[min_cost_idx], torch.tensor(dynamic_label, device=device, dtype=torch.float))
                loss[i] += 2.0 * F.l1_loss(predicts[min_cost_idx], torch.tensor(dynamic_label, device=device, dtype=torch.float))

            loss[i] += F.nll_loss(group_scores.unsqueeze(0), torch.tensor([min_cost_idx], device=device))

            t = np.array(predicts[min_cost_idx].tolist())

            utils.other_errors_put('set_MR_mincost', np.min(utils.get_dis_point_2_points(gt_points[-1], t)) > 2.0)
            utils.other_errors_put('set_minFDE_mincost', np.min(utils.get_dis_point_2_points(gt_points[-1], t)))

        predict = np.array(predicts[argmin].tolist())

        set_predict_ans_points = predict.copy()
        li = []
        for point in set_predict_ans_points:
            li.append((point, scores[np.argmin(utils.get_dis_point_2_points(point, goals_2D))]))  # 每个经过set_predict的目标点，与其对应的初始点的分数
        li = sorted(li, key=lambda x: -x[1])
        set_predict_ans_points = np.array([each[0] for each in li])  # 优化之后的点
        mapping[i]['set_predict_ans_points'] = set_predict_ans_points  # shape(6, 2)

        if args.argoverse:
            utils.other_errors_put('set_MR_pred', np.min(utils.get_dis_point_2_points(gt_points[-1], predict)) > 2.0)
            utils.other_errors_put('set_minFDE_pred', np.min(utils.get_dis_point_2_points(gt_points[-1], predict)))
