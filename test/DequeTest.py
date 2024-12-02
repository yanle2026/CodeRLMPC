import numpy as np
from scipy.stats import truncnorm, norm
from collections import deque

# # 创建一个空的deque
# fifo_queue = deque()
#
# # 向队列添加元素（右侧）
# fifo_queue.append(1)
# fifo_queue.append(2)
# fifo_queue.append(3)
#
# # 从队列中取出元素（左侧）
# print(fifo_queue.popleft())  # 输出: 1
# print(fifo_queue.popleft())  # 输出: 2
#
# # 查看队列中的元素
# print(fifo_queue)  # 输出: deque([3])
#
# # 再次从队列中取出元素
# print(fifo_queue.popleft())  # 输出: 3
#
# # 队列为空时，尝试取出元素会抛出IndexError
# # print(fifo_queue.popleft())  # 抛出: IndexError


def truncated_normal_generator(mean, std_dev, lower, upper):
    """
    创建一个生成二维截断高斯分布随机样本的生成器
    :param mean: 均值 μ，形状为 (2,) 的数组
    :param std_dev: 标准差 σ，形状为 (2,) 的数组
    :param lower: 截断下界，形状为 (2,) 的数组
    :param upper: 截断上界，形状为 (2,) 的数组
    :return: 每次生成一个二维截断高斯分布的随机样本
    """
    # 转换截断区间到标准正态范围
    a_x, b_x = (lower[0] - mean[0]) / std_dev[0], (upper[0] - mean[0]) / std_dev[0]
    a_y, b_y = (lower[1] - mean[1]) / std_dev[1], (upper[1] - mean[1]) / std_dev[1]

    # 创建截断分布
    dist_x = truncnorm(a_x, b_x, loc=mean[0], scale=std_dev[0])
    dist_y = truncnorm(a_y, b_y, loc=mean[1], scale=std_dev[1])

    # 每 next 一次，生成一个样本点 [x, y]
    while True:
        yield np.array([dist_x.rvs(), dist_y.rvs()])


# 创建外界干扰生成器
mean = [1, 1]  # 均值
std_dev = [0.01, 0.02]  # 标准差
lower = [0, 0]  # 下界
upper = [2, 2]  # 上界
sample_generator = truncated_normal_generator(mean, std_dev, lower, upper)
# 创建一个空的二维deque
two_dimensional_deque = deque()

# 初始化二维deque
for i in range(3):
    two_dimensional_deque.append(next(sample_generator))

# 打印二维deque
for row in two_dimensional_deque:
    print(row)
