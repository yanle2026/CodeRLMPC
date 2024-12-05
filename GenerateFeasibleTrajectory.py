import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class GenerateFeasibleTrajectory:
    def __init__(self, T=10, StateBound=GRB.INFINITY, DecisionBound=5):
        # 建模相关变量
        self.T = T
        self.StateBound = StateBound
        self.DecisionBound = DecisionBound
        self.model = None
        self.x1 = None
        self.x2 = None
        self.u = None

        self.noise_constrs = None
        self.build_model()
        # 当前迭代的整个轨迹
        self.X = []
        self.U = []
        self.TerminalQ = deque()
        self.Points = 0 # 当前轨迹中状态点的个数
        # 安全集
        self.SSX = []
        self.SSU = []
        self.SSQ = []

    def build_model(self):
        """
        1. 系统模型：
            x1_t+1 = x1_t + x2_t
            x2_t+1 =        x2_t + u_t
        2. 成本矩阵 R 和 Q 均为单位阵
        """
        # 创建模型
        self.model = gp.Model("ControlOptimization")
        self.model.setParam('OutputFlag', 0)
        # 定义变量
        self.x1 = self.model.addVars(self.T+1, lb=-self.StateBound, ub=self.StateBound, name="x1")  # 状态1
        self.x2 = self.model.addVars(self.T+1, lb=-self.StateBound, ub=self.StateBound, name="x2")  # 状态2
        self.u = self.model.addVars(self.T, lb=-self.DecisionBound, ub=self.DecisionBound, name="u")  # 输入

        # 添加初始状态占位约束
        self.model.addConstr(self.x1[0] == 0, name="initial_x1")
        self.model.addConstr(self.x2[0] == 0, name="initial_x2")

        # 添加状态方程约束
        self.noise_constrs = []
        for t in range(self.T):
            c1 = self.model.addConstr(self.x1[t + 1] == self.x1[t] + self.x2[t], name=f"state1_{t}")
            c2 = self.model.addConstr(self.x2[t + 1] == self.x2[t] + self.u[t], name=f"state2_{t}")
            self.noise_constrs.append((c1, c2))

        self.model.update()

        # 定义目标函数
        self.model.setObjective(gp.quicksum(self.x1[t] ** 2 + self.x2[t] ** 2 + self.u[t] ** 2 for t in range(self.T))
                                +self.x1[self.T] ** 2 + self.x2[self.T] ** 2, GRB.MINIMIZE)

    def optimize_control(self, x1_0, x2_0):
        times = 0
        x1 = x1_0
        x2 = x2_0
        while times<1:
            # 更新初始状态约束
            self.model.getConstrByName("initial_x1").rhs = x1
            self.model.getConstrByName("initial_x2").rhs = x2

            # 优化模型
            self.model.optimize()
            self.model.write('DoubleIntegratorWithNoise.lp')

            if self.model.status == GRB.OPTIMAL:
                self.U.append(self.u[0].X)
                self.X.append([self.x1[0].X, self.x2[0].X])
                self.Points += 1

                self.model.write('DoubleIntegratorWithNoise.lp')
                x1 = self.x1[1].X
                x2 = self.x2[1].X
            else:
                raise RuntimeError("Optimization was not successful. Model status:", self.model.status)

            print("Optimized u values:", self.U[-1], "\nAfter x values[", x1, x2, "]")
            if abs(x1)<1e-5 and abs(x2)<1e-5:
                times+=1
        self.X.append([self.x1[1].X, self.x2[1].X])
        self.SSX.append(np.array(self.X))
        self.SSU.append(np.array(self.U))
        self.refresh_SSQ()

    def plot_trajectory(self):
        # 绘制 optimizer.X1 和 optimizer.X2 的折线图
        plt.figure()
        x1 = [row[0] for row in self.SSX[-1]]
        x2 = [row[1] for row in self.SSX[-1]]
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

    def plot_state_value(self):
        # 绘制 optimizer.X1 和 optimizer.X2 的折线图
        plt.figure(figsize=(10, 6))
        plt.plot([row[0] for row in self.X], label='X1')
        plt.plot([row[1] for row in self.X], label='X2')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('States changes of X1 and X2')
        plt.legend()
        plt.show()

    def refresh_SSQ(self):
        self.TerminalQ.append(0)
        self.X = [sum(x**2 for x in row) for row in self.X]
        self.U = [row**2 for row in self.U]
        self.X.pop()
        while self.Points:
            self.TerminalQ.appendleft(self.X[-1] + self.U[-1] + self.TerminalQ[0])
            self.U.pop()
            self.X.pop()
            self.Points -= 1
        self.SSQ.append(np.array(self.TerminalQ))
        self.TerminalQ.clear()

    def save_statics(self):
        # 将训练所得的历史数据保存为文件
        # 对于大型数据集和数值计算，NumPy数组在内存占用和运行速度方面通常优于Python列表。
        # 但是，如果数据类型不固定或者需要进行频繁的插入和删除操作，Python列表可能更合适。
        np.savez('FeasibleX.npz', *self.SSX)
        np.savez('FeasibleU.npz', *self.SSU)
        np.savez('FeasibleQ.npz', *self.SSQ)


if __name__ == "__main__":
    # 参数设置
    T = 10
    StateBound = GRB.INFINITY
    DecisionBound = 5
    x1_0 = 200.0
    x2_0 = -200.0
    optimizer = GenerateFeasibleTrajectory(T, StateBound, DecisionBound)

    for _ in range(5):
        optimizer.optimize_control(x1_0, x2_0)
        x1_0+=1
        x2_0+=1

    optimizer.save_statics()
    optimizer.plot_trajectory()