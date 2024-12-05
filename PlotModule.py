import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(X, iteration):
    # 绘制 optimizer.X1 和 optimizer.X2 的折线图
    plt.figure()
    x1 = [row[0] for row in X[iteration]]
    x2 = [row[1] for row in X[iteration]]
    plt.scatter(x1, x2, s=5, c='blue', marker='o')
    plt.plot(x1, x2, 'm')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Last iteration of States trajectory')
    # 标记起点和终点
    plt.scatter([x1[0]], [x2[0]], color='red', s=5, label='Start Point')  # 起点标记为红色
    plt.scatter([x1[-1]], [x2[-1]], color='green', s=5, label='End Point')  # 终点标记为绿色
    # 显示网格（可选）
    plt.grid(True)
    # 添加图例
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载保存的数组
    loaded_X = np.load('FeasibleX.npz')
    # 将加载的数组转换回原始的结构（列表形式）
    X = [loaded_X[key] for key in loaded_X.files]
    plot_trajectory(X, 0)