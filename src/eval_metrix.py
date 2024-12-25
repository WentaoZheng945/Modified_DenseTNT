import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from av2.map.map_api import ArgoverseStaticMap, DrivableAreaMapLayer
from av2.map.drivable_area import DrivableArea

def argoverse2_get_map_instance(instance_dir):
    log_map_dirpath = Path(instance_dir)
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    if not len(vector_data_fnames) == 1:
        raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
    vector_data_fname = vector_data_fnames[0]
    map = ArgoverseStaticMap.from_json(vector_data_fname)
    return map

def points_in_drivable_area(map, points_xyz):
    # for drivable_area in map.vector_drivable_areas.values():
    #     coors = drivable_area.xyz
    #     Coors.append(coors)
    drivable_areas = list(map.vector_drivable_areas.values())
    drivable_areas_map = DrivableAreaMapLayer.from_vector_data(drivable_areas)

    raster_values = drivable_areas_map.get_raster_values_at_coords(points_xyz, fill_value=0)
    in_drivable_area = raster_values == 1
    return in_drivable_area

def calc_each_scenario_OR(instance_dir, points_xyz):
    # 计算当前场景的预测终点是否超出道路可行驶范围，结果为1/0
    map = argoverse2_get_map_instance(instance_dir)
    in_drivable_area = points_in_drivable_area(map, points_xyz)
    true_count = np.sum(in_drivable_area)
    return 1 if true_count > 0 else 0

def eval_OR(data_dir, temp_file_dir):
    # files = []
    # for each_dir in data_dir:
    #     print(each_dir)
    #     root, dirs, cur_files = os.walk(each_dir).__next__()
    #     files.extend([os.path.join(each_dir, file) for file in dirs])  # 54931
    # print(files[:5])

    # pbar = tqdm(total=len(files))
    pickle_file = open(os.path.join(temp_file_dir, 'predict_result'), 'rb')
    predict_result = pickle.load(pickle_file)
    pickle_file.close()
    # print(len(predict_result))
    pbar = tqdm(total=len(predict_result))

    count = 0
    file_name_list = []
    for name, pre_traj in predict_result.items():
        instance_dir = os.path.join(data_dir[0], name)
        points_xyz = pre_traj[:, -1, :]
        temp = calc_each_scenario_OR(instance_dir, points_xyz)
        if temp == 0:
            file_name_list.append(name)
        count += temp
        pbar.update(1)

    count = 1 - count/len(predict_result)
    print("\033[31m" + f'OR: {count}' + "\033[0m")
    pbar.close()

    pickle_file = open(os.path.join(temp_file_dir, 'OR'), 'wb')
    pickle.dump(file_name_list, pickle_file)
    pickle_file.close()


def get_speed_and_accer(position_series, rate):
    velocity = np.zeros((6, 60, 2)).astype(float)
    acc = np.zeros((6, 60, 2)).astype(float)
    jerk = np.zeros((6, 60, 2)).astype(float)

    # 差分计算速度（由前后两个点差分得到——中心差分）
    for i in range(60):
        if i == 0:
            velocity[:, i, :] = (position_series[:, i+2, :] - position_series[:, i, :]) / float(2 * rate)
        elif i == 60 - 1:
            velocity[:, i, :] = (position_series[:, i, :] - position_series[:, i-2, :]) / float(2 * rate)
        else:
            velocity[:, i, :] = (position_series[:, i+1, :] - position_series[:, i-1, :]) / float(2 * rate)

    # 差分计算加速度（由前后两个点差分得到）
    for i in range(60):
        if i == 0:
            acc[:, i, :] = (velocity[:, i+2, :] - velocity[:, i, :]) / float(2 * rate)
        elif i == 60 - 1:
            acc[:, i, :] = (velocity[:, i, :] - velocity[:, i-2, :]) / float(2 * rate)
        else:
            acc[:, i, :] = (velocity[:, i+1, :] - velocity[:, i-1, :]) / float(2 * rate)

    # 差分计算加加速度（中心差分）
    for i in range(60):
        if i == 0:
            jerk[:, i, :] = (acc[:, i+2, :] - acc[:, i, :]) / float(2 * rate)
        elif i == 60 - 1:
            jerk[:, i, :] = (acc[:, i, :] - acc[:, i-2, :]) / float(2 * rate)
        else:
            jerk[:, i, :] = (acc[:, i+1, :] - acc[:, i-1, :]) / float(2 * rate)

    return velocity, acc, jerk

def analysis_each_dem_each_scenario(dem_array, limit):
    """
    分析某一场景某一维度的异常值比例
    Args:
        dem_array: dem_array为ndarray，shape(6, 60), 某一维度的信息，比如横向加速度
        limit: 某一维度的阈值， list

    Returns:该维度中异常值最低的那个模态的异常值帧数
    """
    counts = np.sum((dem_array < limit[0]) | (dem_array > limit[1]), axis=1)
    return min(counts)

def calc_abnormal_value(temp_file_dir):
    pickle_file = open(os.path.join(temp_file_dir, 'predict_result'), 'rb')
    predict_result = pickle.load(pickle_file)
    pickle_file.close()
    # pbar = tqdm(total=len(predict_result))
    lat_acc_rate, lon_acc_rate, lat_jerk_rate, lon_jerk_rate = [0 for _ in range(4)]
    for name, pre_traj in predict_result.items():
        velocity, acc, jerk = get_speed_and_accer(pre_traj, 0.1)
        # x为lat, y为lon
        for i in range(2):
            if i == 0:
                lat_acc_rate += analysis_each_dem_each_scenario(acc[:, :, i], [-8, 5])
                lat_jerk_rate += analysis_each_dem_each_scenario(jerk[:, :, i], [-15, 15])
            else:
                lon_acc_rate += analysis_each_dem_each_scenario(acc[:, :, i], [-8, 5])
                lon_jerk_rate += analysis_each_dem_each_scenario(jerk[:, :, i], [-15, 15])
    lat_acc_rate = lat_acc_rate / (len(predict_result) * 60)
    lon_acc_rate = lon_acc_rate / (len(predict_result) * 60)
    lat_jerk_rate = lat_jerk_rate / (len(predict_result) * 60)
    lon_jerk_rate = lon_jerk_rate / (len(predict_result) * 60)
    print('异常值比例：')
    print(
        '横向加速度:',lat_acc_rate,
        '纵向加速度:',lon_acc_rate,
        '横向加加速度:',lat_jerk_rate,
        '纵向加加速度:',lon_jerk_rate,
    )


def jerkAnalysis(jerk):
    '''离群点分析——极端突变值比例与符号反转频率
    :param jerk          : 传入的加加速度张量[nums_scene, length_scene]

    :return jerk_single  : 平均每个场景包含的极端突变值比例
    '''
    jerk_table = [0] * jerk.shape[0]  # 数组哈希表，每个下标对应一个场景，元素值代表该场景中极端突变值的帧数
    index_ = np.where(jerk > 15)  # 返回一个二维数组[行索引，列索引]
    jerk_ind = np.concatenate((index_[0].reshape([-1, 1]), index_[1].reshape([-1, 1])), axis=1)  # 离群点坐标[scene_id, frame_id]
    for i in jerk_ind:
        jerk_table[i[0]] += 1
    jerk_single = 0  # 平均每个场景的极端突变值比例
    if sum(jerk_table):
        jerk_single = np.mean([round(i / jerk.shape[1], 4) for i in jerk_table])
    return jerk_single


if __name__ == '__main__':
    data_dir = [r'/root/autodl-tmp/data/val']
    temp_file_dir = r'/root/zwt/DenseTNT/argoverse2.densetnt.1/temp_file/'
    # data_dir = [r'/media/zwt/新加卷/Argoverse_2/val']
    eval_OR(data_dir, temp_file_dir)
    # calc_abnormal_value(temp_file_dir)