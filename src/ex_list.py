import os
import zlib
import pickle

def get_ex_list(file_path):
    pickle_file = open(file_path, 'rb')
    ex_list = pickle.load(pickle_file)
    # self.ex_list = self.ex_list[len(self.ex_list) // 2:]
    pickle_file.close()
    return ex_list

def parse_ex_list(ex_list):
    print('Length of training dataset:', len(ex_list))  # 54926
    for idx, value in enumerate(ex_list):
        if idx == 0:
            return get_idx(ex_list, idx)
        else:
            return None


def get_idx(ex_list, idx):
    # file = self.ex_list[idx]
    # pickle_file = open(file, 'rb')
    # instance = pickle.load(pickle_file)
    # pickle_file.close()

    data_compress = ex_list[idx]
    instance = pickle.loads(zlib.decompress(data_compress))
    return instance

if __name__ == '__main__':
    file_path = r'/home/zwt/thesis/DenseTNT/DenseTNT-argoverse2/argoverse2.densetnt.1/temp_file/ex_list'
    ex_list = get_ex_list(file_path)
    tmp = parse_ex_list(ex_list)
    print(type(tmp))
    print(tmp.keys())
    for key, value in tmp.items():
        print('key:', key, ' ', 'type:', type(value))
        if key == 'matrix':
            print(value.shape)
            print(value)
            # print(len(value))
            # # print(type(value[0]))
            # for i in range(len(value)):
            #     print(value[i])


"""
ex_list为一个list，每个元素为一个场景，数据结构为dict

    args:
        cent_x: 原始笛卡尔坐标
        cent_y：原始笛卡尔坐标
        angle：90-heading（弧度）
        normalizer：转换器，normalizer(origin_x, origin_y)转换成新的坐标
        goals_2D：ndarray 车道边界的点，拆成很细的颗粒度
        goals_2D_labels：ndarray 距离终点最近的车道边界点的索引
        stage_one_label：终点所在车道多边形的索引
        matrix：ndarray 地图和场景的编码信息
        labels：ndarray 目标车未来所有帧坐标的真值（新坐标系下）
        gt_trajectory_global_coordinates：ndarray 目标车未来所有帧坐标的真值（原始笛卡尔坐标系下）
        polyline_spans： list 存放了每个场景中，每辆车对应的vector的开始索引和结束索引，以及地图信息的开始索引和结束索引
        labels_is_valid： ndarray 每个都是合法的
        eval_time: 60
        agents：所有车前50帧的坐标信息
        map_start_polyline_idx： polyline_spans中地图信息开始的索引
        polygons：list 里面每个元素都是一条车道线的polyline
        file_name：场景名称
        trajs：跟agents一样
        vis_lanes：与polygons一致
    
    eval：
        file_name:场景名
        track_id:轨迹id
        gt_future_track:真实未来轨迹
        pr_future_track:预测未来轨迹
"""
# TODO 重写argoverse2_get_instance_eval函数，使其兼容上面的内容
# TODO 记录预测后的轨迹
# TODO 重写预测后可视化内容