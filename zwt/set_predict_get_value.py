import numpy as np

# 定义一个辅助函数来计算两点之间的欧氏距离
def get_dis_point(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# 定义 _set_predict_get_value 函数，它接受目标点数组、分数数组、选定点数组和一个可选的 kwargs 字典
def _set_predict_get_value(goals_2D, scores, selected_points, kwargs=None):
    # 获取目标点的数量
    n = goals_2D.shape[0]
    # 初始化一个零向量来存储单个点的坐标
    point = np.zeros(2, dtype=np.float32)
    # 初始化 MR 比率为 1.0
    MRratio = 1.0

    # 如果 kwargs 不是 None 并且包含 'set_predict-MRratio' 键，则更新 MRratio 的值
    if kwargs is not None and 'set_predict-MRratio' in kwargs:
        MRratio = float(kwargs['set_predict-MRratio'])

    # 初始化总价值为 0
    value = 0.0
    # 遍历每个目标点
    for i in range(n):
        # 将第 i 个目标点的坐标存储到 point 向量中
        point[0], point[1] = goals_2D[i, 0], goals_2D[i, 1]

        # 初始化当前目标点的总和为 0
        sum = 0.0
        # 将分数乘以 1000 并转换为整数
        t_int = int(scores[i] * 1000)

        # 根据分数的整数表示来确定计数 cnt
        if t_int > 10:
            cnt = 3
        elif t_int > 5:
            cnt = 2
        else:
            cnt = 1

        # 计算步长 stride
        t_float = cnt
        stride = 1.0 / t_float
        # 计算起始坐标 s_x 和 s_y
        s_x = point[0] - 0.5 + stride / 2.0
        s_y = point[1] - 0.5 + stride / 2.0

        # 遍历每个网格点
        for a in range(int(t_float)):
            for b in range(int(t_float)):
                # 计算网格点的坐标
                x = s_x + a * stride
                y = s_y + b * stride
                # 初始化最小 FDE 为一个很大的数
                minFDE = 10000.0
                # 初始化未命中误差为 1.0
                miss_error = 1.0
                # 遍历每个选定点
                for j in range(selected_points.shape[0]):
                    # 计算网格点与选定点之间的距离
                    t_float = get_dis_point(x, y, selected_points[j, 0], selected_points[j, 1])
                    # 如果找到更小的距离，则更新 minFDE
                    if t_float < minFDE:
                        minFDE = t_float
                # 如果 minFDE 小于等于 2.0，则未命中误差为 0
                if minFDE <= 2.0:
                    miss_error = 0.0
                # 将计算的 minFDE 和 miss_error 加权求和并加到 sum 中
                sum += minFDE * (1.0 - MRratio) + miss_error * MRratio
        # 将 sum 除以网格点的总数
        sum /= (int(t_float) ** 2)
        # 将加权后的 sum 加到总价值 value 中
        value += scores[i] * sum

    # 返回计算出的总价值
    return value

# # 示例用法：
# goals_2D = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
# scores = np.array([0.5, 0.8], dtype=np.float32)
# selected_points = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]], dtype=np.float32)
# kwargs = {'set_predict-MRratio': 0.5}
#
# # 调用函数并打印结果
# result = _set_predict_get_value(goals_2D, scores, selected_points, kwargs)
# print("Calculated value:", result)
def test():
    a = 5.0
    count = 0
    for i in range(int(a)):
        count+=1
        a = 3
    return a


if __name__ == '__main__':
    result = test()
    print(result)