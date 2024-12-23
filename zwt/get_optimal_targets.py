import numpy as np
import random
import time
import math


def _get_optimal_targets(goals_2D, scores, file_name, objective, num_step, cnt_sample, MRratio, opti_time, kwargs):
    n = goals_2D.shape[0]
    m = 0
    threshold = 0.001
    ans_points = np.zeros((6, 2), dtype=np.float32)
    nxt_points = np.zeros((6, 2), dtype=np.float32)
    pred_probs = np.zeros(6, dtype=np.float32)

    best_expectation = 10000.0
    best_points = np.zeros((6, 2), dtype=np.float32)

    start_time = time.time()

    if opti_time < 100.0:
        num_step = 1000000

    # Filtering goals based on the scores
    # 几乎有可能的都过滤出来，最终有m个，替换到goals_2D的前m个
    for i in range(n):
        if scores[i] >= threshold:
            goals_2D[m, 0] = goals_2D[i, 0]
            goals_2D[m, 1] = goals_2D[i, 1]
            scores[m] = scores[i]
            m += 1
    if m == 0:
        print('Warning: m == 0')
        m = n

    n = m

    # Initializing the ans_points with random goals
    # 随机生成初始化样本
    for j in range(6):  # 只要6个
        t_int = random.randint(0, n - 1)
        ans_points[j, 0] = goals_2D[t_int, 0]
        ans_points[j, 1] = goals_2D[t_int, 1]

    # Calculating initial expectation
    expectation = get_value(goals_2D, scores, ans_points, n, objective, cnt_sample, MRratio, kwargs)

    # Main optimization loop
    for step in range(num_step):
        if (time.time() - start_time) >= opti_time:
            break

        ratio = step / num_step
        lr = math.exp(-(ratio * 2))

        # Copy current ans_points to nxt_points
        for j in range(6):
            nxt_points[j, 0] = ans_points[j, 0]
            nxt_points[j, 1] = ans_points[j, 1]

        # Random operation to adjust points
        op = random.randint(0, 0)
        if op == 0:
            while True:
                ok = 0
                for j in range(6):
                    if random.random() < 0.3:
                        nxt_points[j, 0] += random.uniform(-lr, lr)
                        nxt_points[j, 1] += random.uniform(-lr, lr)
                        ok = 1
                if ok:  # 只要上面执行一次，就说明nxt_points与之前的ans_points不同了
                    break

        # Calculate next expectation
        nxt_expectation = get_value(goals_2D, scores, nxt_points, n, objective, cnt_sample, MRratio, kwargs)

        # Decide whether to accept new points
        go = 0
        if nxt_expectation < expectation:
            go = 1
        else:
            fire_prob = 0.01
            if random.random() < fire_prob:
                go = 1

        if go:
            expectation = nxt_expectation
            for j in range(6):
                ans_points[j, 0] = nxt_points[j, 0]
                ans_points[j, 1] = nxt_points[j, 1]

        # Update the best solution found so far
        if expectation < best_expectation:
            best_expectation = expectation
            for j in range(6):
                best_points[j, 0], best_points[j, 1] = ans_points[j, 0], ans_points[j, 1]

    # Final comparison to find the best solution
    min_expectation = 10000.0
    argmin = 0
    for j in range(6):
        for k in range(6):
            nxt_points[k, 0], nxt_points[k, 1] = best_points[j, 0], best_points[j, 1]

        t = get_value(goals_2D, scores, nxt_points, n, 'minFDE', cnt_sample, MRratio, kwargs)
        pred_probs[j] = 1.0 - get_value(goals_2D, scores, nxt_points, n, objective, cnt_sample, MRratio, kwargs)

        if t < min_expectation:
            min_expectation = t
            argmin = j

    return best_expectation, best_points, argmin, pred_probs

def get_dis_point(x, y):
    return np.sqrt(x**2 + y**2)

def get_value(goals_2D, scores, selected_points, n, objective, cnt_sample, MRratio, kwargs):
    value = 0.0
    cnt_len = 0
    sum = 0.0
    minFDE = 10000.0
    miss_error = 10.0
    stride = 0.0
    s_x, s_y = 0.0, 0.0
    cnt = 0
    t_int = 0
    point = np.zeros(2, dtype=np.float32)

    # Calculate the square root of cnt_sample to find cnt_len
    for i in range(100):
        if i * i == cnt_sample:
            cnt_len = i  # 3*3
    if cnt_len == 0:
        raise ValueError('cnt_sample is not a perfect square')

    # Main loop to calculate value
    for i in range(n):
        point[0], point[1] = goals_2D[i, 0], goals_2D[i, 1]
        sum = 0.0
        t_int = int(scores[i] * 1000)  # 把概率放大1000倍

        if t_int > 10:
            cnt = cnt_len * 3
        elif t_int > 5:
            cnt = cnt_len * 2
        else:
            cnt = cnt_len

        t_float = cnt  # 边长为3
        stride = 1.0 / t_float  # 步长为1/3

        s_x = point[0] - 0.5 + stride / 2.0
        s_y = point[1] - 0.5 + stride / 2.0

        for a in range(cnt):
            for b in range(cnt):
                # Using a grid-based approach for the (x, y) values
                x = s_x + a * stride
                y = s_y + b * stride
                minFDE = 10000.0
                miss_error = 10.0

                # Calculate the minimum distance from the selected points
                for j in range(6):
                    t_float = get_dis_point(x - selected_points[j, 0], y - selected_points[j, 1])
                    if t_float < minFDE:
                        minFDE = t_float

                if minFDE <= 2.0:
                    miss_error = 0.0
                sum += minFDE * (1.0 - MRratio) + miss_error * MRratio

        sum /= cnt * cnt
        value += scores[i] * sum

    return value