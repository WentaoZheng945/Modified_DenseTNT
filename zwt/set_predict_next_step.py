import numpy as np

# 定义一个生成指定范围内随机数的函数
def get_rand(a, b):
    return np.random.uniform(a, b)

# 假设这是之前提供的 _set_predict_get_value 函数的实现
def _set_predict_get_value(goals_2D, scores, selected_points, kwargs):
    pass

def _set_predict_next_step(goals_2D, scores, selected_points, lr, kwargs):
    # 初始化变量
    step = 0
    j = 0
    ok = 0
    num_step = 100  # 设置迭代次数的默认值
    nxt_expectation = _set_predict_get_value(goals_2D, scores, selected_points, kwargs)  # 计算初始的期望值
    best_expectation = nxt_expectation  # 初始化最佳期望值为初始期望值
    nxt_points = np.zeros((6, 2), dtype=np.float32)  # 初始化下一个点集为零矩阵
    best_points = selected_points.copy()  # 复制初始点集作为最佳点集

    # 如果 kwargs 字典中包含 'dynamic_label-double' 键，则将迭代次数设置为 200
    if kwargs is not None and 'dynamic_label-double' in kwargs:
        num_step = 200

    # 进行 num_step 次迭代
    for step in range(num_step):
        for j in range(6):  # 对于每个点
            nxt_points[j, 0] = selected_points[j, 0]  # 复制 x 坐标
            nxt_points[j, 1] = selected_points[j, 1]  # 复制 y 坐标

        # 对于每次迭代，尝试随机扰动点集
        while True:
            ok = 0  # 初始化 ok 为 0，用于标记是否进行了扰动
            for j in range(6):  # 对于每个点
                if get_rand(0.0, 1.0) < 0.5:  # 有 50% 的概率进行扰动
                    nxt_points[j, 0] += get_rand(-lr, lr)  # 随机扰动 x 坐标
                    nxt_points[j, 1] += get_rand(-lr, lr)  # 随机扰动 y 坐标
                    ok = 1  # 标记为进行了扰动
            if ok:  # 如果进行了扰动，则跳出循环
                break

        # 计算扰动后的期望值
        nxt_expectation = _set_predict_get_value(goals_2D, scores, nxt_points, kwargs)
        # 如果扰动后的期望值更小（即更好），则更新最佳期望值和最佳点集
        if nxt_expectation < best_expectation:
            best_expectation = nxt_expectation
            for j in range(6):
                best_points[j, 0], best_points[j, 1] = nxt_points[j, 0], nxt_points[j, 1]

    # 返回最佳期望值和对应的最佳点集
    return best_expectation, best_points

# Example usage:
goals_2D = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
scores = np.array([0.5, 0.8, 0.3, 0.9, 0.2, 0.6], dtype=np.float32)
selected_points = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1], [7.1, 8.1], [9.1, 10.1], [11.1, 12.1]], dtype=np.float32)
lr = 0.1
kwargs = {'dynamic_label-double': True}

best_expectation, best_points = _set_predict_next_step(goals_2D, scores, selected_points, lr, kwargs)
print("Best Expectation:", best_expectation)
print("Best Points:\n", best_points)