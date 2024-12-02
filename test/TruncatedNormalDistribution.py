import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
# 定义截断高斯分布生成器
def truncated_normal_generator(mean, std_dev, lower, upper, size=1):
    """
    创建一个生成二维截断高斯分布随机样本的生成器
    :param mean: 均值 μ，形状为 (2,) 的数组
    :param std_dev: 标准差 σ，形状为 (2,) 的数组
    :param lower: 截断下界，形状为 (2,) 的数组
    :param upper: 截断上界，形状为 (2,) 的数组
    :param size: 总样本数量
    :return: 每次生成一个二维截断高斯分布的随机样本
    """
    # 转换截断区间到标准正态范围
    a_x, b_x = (lower[0] - mean[0]) / std_dev[0], (upper[0] - mean[0]) / std_dev[0]
    a_y, b_y = (lower[1] - mean[1]) / std_dev[1], (upper[1] - mean[1]) / std_dev[1]

    # 创建截断分布
    dist_x = truncnorm(a_x, b_x, loc=mean[0], scale=std_dev[0])
    dist_y = truncnorm(a_y, b_y, loc=mean[1], scale=std_dev[1])

    # for _ in range(size):
    #     # 生成一个样本点 [x, y]
    #     yield np.array([dist_x.rvs(), dist_y.rvs()])
    while True:
        yield np.array([dist_x.rvs(), dist_y.rvs()])

if __name__ == '__main__':
    # 示例参数
    mean = [0, 0]  # 均值
    std_dev = [0.01, 0.02]  # 标准差
    lower = [-1, -1]  # 下界
    upper = [1, 1]  # 上界
    size = 10000  # 样本大小

    # 创建生成器
    # sample_generator = truncated_normal_generator(mean, std_dev, lower, upper, size)
    sample_generator = truncated_normal_generator(mean, std_dev, lower, upper)
    samples_list = [next(sample_generator) for _ in range(size)]
    print(samples_list)
    # 生成样本并转换为数组
    # samples = np.array(list(sample_generator))

    # 可视化二维截断高斯分布
    # plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    # plt.title('2D Truncated Normal Distribution')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.axis('equal')
    # plt.show()
