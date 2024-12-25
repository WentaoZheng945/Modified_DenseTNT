import os
import math
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import MultipleLocator

time_begin = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
origin_point = None
origin_angle = None
traj_last = None
visualize_num = 0

def is_stationary_vehicle(full_trajectory):
    dis = np.sqrt((full_trajectory[-1][0] - full_trajectory[0][0])**2+(full_trajectory[-1][1] - full_trajectory[0][1])**2)
    if dis < 5:
        return True
    else:
        return False

def calc_polygon_point_xy(x, y, heading, width=1.5, length=2.0):
    # points = np.array([[-length/2, -width/2],
    #                    [length/2, -width/2],
    #                    [length/2, width/2],
    #                    [-length/2, width/2],])
    # R = np.array([[np.cos(heading), -np.sin(heading)],
    #               [np.sin(heading), np.cos(heading)]])
    # vehicle_xy = np.dot(points, R) + np.array([x, y])
    # return corners


    # 相对中心的四个角点偏移
    offsets = np.array([
        [length / 2, width / 2],  # 左前
        [length / 2, -width / 2],  # 右前
        [-length / 2, -width / 2],  # 左后
        [-length / 2, width / 2]  # 右后
    ])

    # 旋转角度（弧度）
    # theta_rad = np.radians(heading)

    # 旋转矩阵
    rotation_matrix = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])

    # 计算全局坐标
    vehicle_xy = (rotation_matrix @ offsets.T).T + np.array([x, y])

    return vehicle_xy

def load_full_info(temp_file_dir):
    pickle_file = open(os.path.join(temp_file_dir, 'full_info_test'), 'rb')
    full_info = pickle.load(pickle_file)
    pickle_file.close()
    return full_info

def add_argument(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    # Required parameters
    parser.add_argument("--do_train", default=False, help="whether to run training")
    parser.add_argument("--do_eval", default=True, help="whether to run validation")
    parser.add_argument("--do_test", default=False, help="whether to run testing")
    parser.add_argument("--debug", default=False, help="whether to debug")
    parser.add_argument("--add_prefix", default=None, help="add prefix")
    parser.add_argument("--temp_file_dir", default='/home/zwt/thesis/DenseTNT/trained_model/argoverse2.densetnt.1/temp_file/', help="temp file dir")
    parser.add_argument("--eval_single", default=False, help="whether to evaluate single agent")
    parser.add_argument("--visualize_num", default=100, type=int)  # 可视化张数
    parser.add_argument("--visualize_percent", default=None, type=float)  # 可视化比例
    parser.add_argument("--view", default='global view', help="Visualization view, global view or vehicle view, default is global View")
    parser.add_argument("--vis_mode", default='multi', help="Visualization mode, multi or single, default is multi")
    parser.add_argument("--log_dir", default='/home/zwt/thesis/DenseTNT/trained_model/argoverse2.densetnt.1/', type=str, help="visualization dir")
    return

def group_by_file_name(full_info):
    grouped_by_file_name = {}
    for key, value in full_info.items():
        file_name = value['file_name']

        if file_name not in grouped_by_file_name:
            grouped_by_file_name[file_name] = []

        grouped_by_file_name[file_name].append(value)
    print("\033[m" + f'Number of scenarios： {len(grouped_by_file_name)}' + "\033[0m")
    return grouped_by_file_name

def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y

def to_origin_coordinate_(points, idx_in_batch, scale=None):
    if np.ndim(points) == 1:
        points[0], points[1] = rotate(points[0] - origin_point[idx_in_batch][0],
                                      points[1] - origin_point[idx_in_batch][1], origin_angle[idx_in_batch])
        if scale is not None:
            points[0] *= scale
            points[1] *= scale
    else:
        for point in points:
            point[0], point[1] = rotate(point[0] - origin_point[idx_in_batch][0],
                                        point[1] - origin_point[idx_in_batch][1], origin_angle[idx_in_batch])
            if scale is not None:
                point[0] *= scale
                point[1] *= scale

def file_name_init(mapping):
    global traj_last, origin_point, origin_angle
    batch_size = len(mapping)

    global origin_point, origin_angle
    origin_point = np.zeros([batch_size, 2])
    origin_angle = np.zeros([batch_size])
    for i in range(batch_size):
        origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
                                                        mapping[i]['angle'])
        origin_angle[i] = -mapping[i]['angle']

def main():
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    full_info = load_full_info(args.temp_file_dir)
    # print(full_info)
    print("\033[m" + f'Number of predicted single agent trajectories： {len(full_info)}' + "\033[0m")

    # TODO检查参数是否正确
    # 两种视角：主车视角(vehicle view)或者全局视角(global view)
    # 两种绘制模式: single or multi
    # 主车视角仅支持单车，全局视角仅支持多车
    if args.view == 'global view':
        assert args.vis_mode == 'multi', 'Global view only supports the plotting of multi mode'
    elif args.view == 'vehicle view':
        assert args.vis_mode == 'single', 'Vehicle view only supports the plotting of single mode'
    else:
        raise ValueError(f'Currently only vehicle view and global view are available, view {args.view} is not available')

    global visualize_num

    # TODO处理序列化pickle
    if args.view == 'global view':
        # 处理
        grouped_by_file_name = group_by_file_name(full_info)
        # TODO绘制的张数设置(添加%)
        # 计算绘制的张数
        if args.visualize_percent is not None:
            args.visualize_num = int(args.visualize_percent * len(grouped_by_file_name))
        else:
            args.visualize_num = args.visualize_num
        # TODO进度条
        pbar = tqdm(total=args.visualize_num)
        for file_name in grouped_by_file_name.keys():
            file_name_init(list(grouped_by_file_name[file_name]))
            # TODO 去看绘图的函数看看哪些地方需要还原，还原回原全局坐标系下
            # TODO 需要设计新的数据结构，来存放一整个场景的数据(仅存放可视化需要的即可)
            # TODO 可视化函数重写
            # prediction, map_info, background
            prediction = {}
            map_info = None
            background = {}
            for idx, value in enumerate(grouped_by_file_name[file_name]):
                # 把车道线还原为全局坐标
                for boundary in value['vis_lanes']:  # 每个元素是ndarray，为左边界或者右边界
                    to_origin_coordinate_(boundary, idx)
                # 车辆历史轨迹
                for track in value['full_trajectory'].values():
                    for attribute in ['past', 'future']:
                        for time_idx in track[attribute].keys():
                            track[attribute][time_idx] = np.array(track[attribute][time_idx])
                            to_origin_coordinate_(track[attribute][time_idx], idx)
                # 预测的候选终点坐标
                to_origin_coordinate_(value['vis_post.goals_2D'], idx)
                # 真实轨迹坐标
                to_origin_coordinate_(value['vis_post.labels'], idx)
                # 预测轨迹
                for pre_traj in value['vis_post.predict_trajs']:
                    to_origin_coordinate_(pre_traj, idx)
                if value['track_id'] not in prediction:
                    prediction[value['track_id']] = {}
                    prediction[value['track_id']].update(dict(
                        vis_post_goals_2D=value['vis_post.goals_2D'],  # ndarray
                        vis_post_labels=value['vis_post.labels'],  # ndarray
                        vis_post_predict_trajs=value['vis_post.predict_trajs'],  # ndarray
                        vis_post_past_trajs=np.array(list(value['full_trajectory'][value['track_id']]['past'].values())),  # ndarray
                        category=value['full_trajectory'][value['track_id']]['category']  # str
                    ))
                map_info = value['vis_lanes']
                for track_id in value['full_trajectory'].keys():
                    if track_id not in list(background.keys()):
                        background[track_id] = {}
                        background[track_id].update(dict(
                            category=value['full_trajectory'][track_id]['category'],  # str
                            trajs = np.array(list(value['full_trajectory'][track_id]['past'].values()) +
                                             list(value['full_trajectory'][track_id]['future'].values()))  # ndarray
                        ))
            # 删除多余背景车
            for track_id in list(prediction.keys()):
                if track_id in background:
                    del background[track_id]

            # 超过规定的张数停止
            visualize_num += 1
            if visualize_num > args.visualize_num:
                break
            plot_scenario_global_view(prediction, map_info, background, args, file_name)
            # 更新进度条
            pbar.update(1)
        pbar.close()
    else:
        pass
    return

def draw_prediction(ax, prediction, add_end=False):
    marker_size_scale = 2  # 图中散点的大小是默认的两倍
    max_x, max_y = -float('inf'), -float('inf')
    min_x, min_y = float('inf'), float('inf')
    target_agent_color, target_agent_edge_color = '#4bad34', '#c5dfb3'  # 绿色
    for track_id, track in prediction.items():
        # 计算朝向
        first_five_elements = track['vis_post_past_trajs'][:5]
        first_five_headings = np.zeros(4)
        for i in range(4):
            first_five_headings[i] = np.arctan2(first_five_elements[i+1][1] - first_five_elements[i][1],
                                                first_five_elements[i+1][0] - first_five_elements[i][0])
        heading = np.mean(first_five_headings)
        # 计算形状
        if track['category'] == 'vehicle':
            vehicle_box = calc_polygon_point_xy(track['vis_post_past_trajs'][0][0], track['vis_post_past_trajs'][0][1], heading, width=1.5, length=2.5)
        else:
            raise ValueError('Unknown category: {}'.format(track['category']))

        # 绘制车辆矩形
        rectangle = patches.Polygon(vehicle_box, closed=True, facecolor=target_agent_color, edgecolor=target_agent_edge_color,
                                    linewidth=2, alpha=1.0, zorder=1)
        ax.add_patch(rectangle)

        # 绘制历史轨迹
        ax.plot(track['vis_post_past_trajs'][:, 0], track['vis_post_past_trajs'][:, 1], linestyle="-", color='darkblue', marker='o',
                alpha=1, linewidth=2, markersize=3, zorder=2)
        # 统计一下绘图范围
        max_x = max(max_x, max(track['vis_post_past_trajs'][:, 0]))
        max_y = max(max_y, max(track['vis_post_past_trajs'][:, 1]))
        min_x = min(min_x, min(track['vis_post_past_trajs'][:, 0]))
        min_y = min(min_y, min(track['vis_post_past_trajs'][:, 1]))

        # 绘制预测车真实轨迹
        function1 = ax.plot(track['vis_post_labels'][:, 0], track['vis_post_labels'][:, 1], linestyle="-", color=target_agent_color, marker='o',
                            alpha=1, linewidth=2, markersize=3, zorder=2, label='Ground truth trajectory')
        # 统计一下绘图范围
        max_x = max(max_x, max(track['vis_post_labels'][:, 0]))
        max_y = max(max_y, max(track['vis_post_labels'][:, 1]))
        min_x = min(min_x, min(track['vis_post_labels'][:, 0]))
        min_y = min(min_y, min(track['vis_post_labels'][:, 1]))
        # 绘制终点特殊标记
        if add_end:
            ax.plot(track['vis_post_labels'][-1, 0], track['vis_post_labels'][-1, 1], markersize=15*marker_size_scale,
                    color=target_agent_color, marker='*', markeredgecolor='black', zorder=1)

        # 绘制预测轨迹
        for each in track['vis_post_predict_trajs']:
            function2 = ax.plot(each[:, 0], each[:, 1], linestyle="-", color='darkorange', marker='o',
                                alpha=1, linewidth=2, markersize=3, zorder=2, label='Predicted trajectory')

            if add_end:
                ax.plot(each[-1, 0], each[-1, 1], markersize=15*marker_size_scale,
                        color='darkorange', marker='*', markeredgecolor='black', zorder=1)
        # 统计一下绘图范围
        max_x = max(max_x, np.max(track['vis_post_predict_trajs'][:, :, 0]))
        max_y = max(max_y, np.max(track['vis_post_predict_trajs'][:, :, 1]))
        min_x = min(min_x, np.min(track['vis_post_predict_trajs'][:, :, 0]))
        min_y = min(min_y, np.min(track['vis_post_predict_trajs'][:, :, 1]))
    return max_x, min_x, max_y, min_y, function1, function2

def draw_background(ax, background, filter=True):
    background_agent_color, background_agent_edge_color = '#0d79e7', '#bcd6ed' # blue
    for track_id, track in background.items():
        if filter:
            # filter从三个维度开始，一个是去除无效类型， 一个是去除时间小于0.5s,
            # 最后是去除车道外静止车辆(目前做法是计算背景车最后一帧和第一帧坐标的距离小于5m TODO 未来考虑和OR指标一样使用原始地图的raster)
            if track['category'] in ['static', 'background', 'construction', 'riderless_bicycle', 'unknown']:
                continue
            if track['trajs'].shape[0] < 5:
                continue
            if is_stationary_vehicle(track['trajs']):
                continue
        first_five_elements = track['trajs'][:5]
        first_five_heading = np.zeros(4)
        for i in range(4):
            first_five_heading[i] = np.arctan2(first_five_elements[i+1][1] - first_five_elements[i][1],
                                               first_five_elements[i+1][0] - first_five_elements[i][0])
        heading = np.mean(first_five_heading)

        # 计算形状
        if track['category'] == 'vehicle':
            vehicle_box = calc_polygon_point_xy(track['trajs'][0][0], track['trajs'][0][1], heading, width=1.5, length=2.5)
        elif track['category'] == 'pedestrian':
            vehicle_box = calc_polygon_point_xy(track['trajs'][0][0], track['trajs'][0][1], heading, width=0.2, length=0.2)
        elif track['category'] == 'motorcyclist':
            vehicle_box = calc_polygon_point_xy(track['trajs'][0][0], track['trajs'][0][1], heading, width=0.5, length=1)
        elif track['category'] == 'cyclist':
            vehicle_box = calc_polygon_point_xy(track['trajs'][0][0], track['trajs'][0][1], heading, width=0.5, length=1)
        elif track['category'] == 'bus':
            vehicle_box = calc_polygon_point_xy(track['trajs'][0][0], track['trajs'][0][1], heading, width=2, length=6)
        else:
            raise ValueError('Unknown category: {}'.format(track['category']))

        # 绘制车辆矩形
        rectangle = patches.Polygon(vehicle_box, closed=True, facecolor=background_agent_color, edgecolor=background_agent_edge_color, linewidth=2, alpha=1.0, zorder=1)
        ax.add_patch(rectangle)
        # ax.text(track[0][0], track[0][1], f"{track['category']}", color='black', fontsize=10, ha='center', va='center')

        # 绘制车辆轨迹
        ax.plot(track['trajs'][:, 0], track['trajs'][:, 1], linestyle="-", color='darkblue', marker='o',
                alpha=1, linewidth=2, markersize=3, zorder=1)
    return

def draw_map(ax, map_info):
    line_width = 2

    # 绘制车道线
    for boundary in map_info:
        points = boundary[:, :2]
        # 地图车道画黑色实线，不做标记，线宽为2，zorder为0，透明度为0.5
        ax.plot(points[:, 0], points[:, 1], linestyle="-", color='black', marker=None,
                markersize=0, alpha=0.5, linewidth=line_width, zorder=0)
        # 填充
        # ax.fill(points[:, 0], points[:, 1], linestyle="-", color='#a5a5a5', alpha=0.5,
        #         linewidth=line_width, zorder=0)
    return

def plot_scenario_global_view(prediction: Dict[str, Dict[str, Union[np.array, str]]],
                              map_info: np.array,
                              background: Dict[str, Dict[str, Union[str, np.ndarray]]],
                              args: argparse.Namespace,
                              file_name: str = None,
                              add_end: bool = False):
    """
    Plot scenario global view
    Args:
        prediction: 所有预测车的轨迹
        map_info: 地图信息
        background: 背景车轨迹信息
        args: argparse.Namespace 参数设置
        file_name: 场景名
        add_end: 是否给终点加修饰

    Returns:

    """
    fig_scale = 1.0  # 原始比例不缩放

    def get_scaled_int(a):
        return round(a * fig_scale)

    # 图形对象，设置长、宽
    fig, ax = plt.subplots(figsize=(get_scaled_int(45), get_scaled_int(45)))

    # TODO 怎么解决候选点的概率问题
    # 实用Reds颜色映射
    # cmap = plt.cm.get_cmap('Reds')
    # v_min = np.log(0.00001)
    # v_max = np.max(scores)
    # scores = np.clip(scores.copy(), a_min=v_min, a_max=v_max)
    # 创建一个颜色映射对象
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(v_min=v_min, v_max=v_max))

    # 画出颜色条
    # cbar = fig.colorbar(sm, ax=ax)
    # cbar.set_label('Score Intensity')

    name = file_name if file_name is not None else ""
    add_end = True

    # 绘制车道线
    draw_map(ax, map_info)
    # 绘制预测车轨迹
    scope = draw_prediction(ax, prediction, add_end=add_end)
    # 绘制背景车轨迹
    draw_background(ax, background)

    # TODO 设置显示范围
    ax.set_xlim(scope[1]-20, scope[0]+20)
    ax.set_ylim(scope[3]-20, scope[2]+20)

    # 设置legend
    # ax.legend(loc=0)  # loc显示在最佳位置
    functions = scope[4] + scope[5]
    fun_labels = [f.get_label() for f in functions]
    ax.legend(functions, fun_labels, loc=0)  # loc显示在最佳位置
    ax.set_title(name)
    ax.set_aspect(1)  # 设置纵横比为1， x和y轴的单位长度相同
    ax.xaxis.set_major_locator(MultipleLocator(4))  # 设置每个轴的刻度间隔为4
    ax.yaxis.set_major_locator(MultipleLocator(4))

    os.makedirs(os.path.join(args.log_dir, 'visualize_' + time_begin), exist_ok=True)
    fig.savefig(os.path.join(args.log_dir, 'visualize_' + time_begin,
                            get_name(args, "visualize" + "_" + name + '.png')), bbox_inches='tight')
    plt.close(fig)

    return

def get_name(args, name='', append_time=False):
    if name.endswith(time_begin):
        return name
    prefix = 'test.' if args.do_test else 'eval.' if args.do_eval and not args.do_train else ''
    prefix = 'debug.' + prefix if args.debug else prefix
    prefix = args.add_prefix + '.' + prefix if args.add_prefix is not None else prefix
    suffix = '.' + time_begin if append_time else ''
    return prefix + str(name) + suffix


if __name__ == '__main__':
    main()