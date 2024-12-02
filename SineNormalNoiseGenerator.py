import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def sine_wave(amplitude=1, frequency=1, phase=0, step_size=0.001):
    """
    无限时间正弦波生成器。

    :param amplitude: 正弦波的幅度
    :param frequency: 正弦波的频率（Hz）
    :param phase: 正弦波的初始相位（弧度）
    :param step_size: 每次迭代增加的时间步长，它的倒数为采样频率
    :return: 正弦波当前时间的值
    """
    t = 0  # 初始化时间为0
    while True:
        # 计算当前时间的正弦波值
        sine_value = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        yield sine_value
        # 增加时间步长
        t += step_size


def truncated_normal(mean=0, std_dev=0.01, lower=-1, upper=1):
    """
    创建一个生成二维截断高斯分布随机样本的生成器

    :param mean: 均值 μ，形状为 (2,) 的数组
    :param std_dev: 标准差 σ，形状为 (2,) 的数组
    :param lower: 截断下界，形状为 (2,) 的数组
    :param upper: 截断上界，形状为 (2,) 的数组
    :return: 每次生成一个二维截断高斯分布的随机样本
    """
    # 转换截断区间到标准正态范围
    a = (lower - mean) / std_dev
    b = (upper - mean) / std_dev
    # 创建截断分布
    dist = truncnorm(a, b, loc=mean, scale=std_dev)
    while True:
        yield dist.rvs()

def noise(amplitude=1, frequency=1, phase=0, step_size=0.01, mean=0, std_dev=0.1, lower=-1, upper=1):
    t = 0  # 初始化时间为0
    # 转换截断区间到标准正态范围
    a = (lower - mean) / std_dev
    b = (upper - mean) / std_dev
    # 创建截断分布
    dist = truncnorm(a, b, loc=mean, scale=std_dev)
    while True:
        # 计算当前时间的正弦波值
        sine_value = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        yield np.array([sine_value + dist.rvs(),sine_value + dist.rvs()])
        # 增加时间步长
        t += step_size

if __name__ == '__main__':
    # 创建生成器
    sine_wave_generator = sine_wave(amplitude=0.1, frequency=3, step_size=0.001)
    truncated_normal_generator = truncated_normal(mean=0, std_dev=0.01, lower=0, upper=1)
    # 获取前1000个生成值
    sine_values = [next(sine_wave_generator) for _ in range(1000)]
    normal_values = [next(truncated_normal_generator) for _ in range(1000)]
    noise_values = sine_values + normal_values

    # 绘制正弦波
    plt.plot(noise_values)
    plt.title("noise values")
    plt.xlabel("Time Steps")
    plt.ylabel("noise values")
    plt.show()
