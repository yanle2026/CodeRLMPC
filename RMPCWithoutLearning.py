import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
from collections import deque

pass  # 生成外界干扰
# def truncated_normal_generator(mean, std_dev, lower, upper):
#     """
#     创建一个生成二维截断高斯分布随机样本的生成器
#     :param mean: 均值 μ，形状为 (2,) 的数组
#     :param std_dev: 标准差 σ，形状为 (2,) 的数组
#     :param lower: 截断下界，形状为 (2,) 的数组
#     :param upper: 截断上界，形状为 (2,) 的数组
#     :return: 每次生成一个二维截断高斯分布的随机样本
#     """
#     # 转换截断区间到标准正态范围
#     a_x, b_x = (lower[0] - mean[0]) / std_dev[0], (upper[0] - mean[0]) / std_dev[0]
#     a_y, b_y = (lower[1] - mean[1]) / std_dev[1], (upper[1] - mean[1]) / std_dev[1]
#
#     # 创建截断分布
#     dist_x = truncnorm(a_x, b_x, loc=mean[0], scale=std_dev[0])
#     dist_y = truncnorm(a_y, b_y, loc=mean[1], scale=std_dev[1])
#
#     # 每 next 一次，生成一个样本点 [x, y]
#     while True:
#         yield np.array([dist_x.rvs(), dist_y.rvs()])
#
#
# # 创建外界干扰生成器
# mean = [0.01, 0.01]  # 均值
# std_dev = [0.001, 0.001]  # 标准差
# lower = [0, 0]  # 下界
# upper = [0.02, 0.02]  # 上界
# sample_generator = truncated_normal_generator(mean, std_dev, lower, upper)
pass  # 初始代码
# # 创建模型
# model = gp.Model("DoubleIntegratorWithNoise")
#
# # 定义时间步数和状态变量
# initial_state = [2, 2]
# T = 10  # 总时间步数
# StateBound = 5   # 状态变量边界
# DecisionBound = 5   # 决策变量边界
# x1 = model.addVars(T, lb=-StateBound, ub=StateBound, name="x1")  # 状态1
# x2 = model.addVars(T, lb=-StateBound, ub=StateBound, name="x2")  # 状态2
# u = model.addVars(T, lb=-DecisionBound, ub=DecisionBound,name="u")  # 输入
# noise = deque()
# for i in range(T):
#     noise.append(next(sample_generator))
#     print(noise)
#     print(noise[i][0])
# # 优化循环
# while True:
# # 初始约束添加
#     for t in range(T-1):
#         c1 = model.addConstr(x1[t+1] == x1[t] + x2[t] + noise[t][0], name=f"state1_{t}")
#         c2 = model.addConstr(x2[t+1] == x2[t] + u[t] + noise[t][1], name=f"state2_{t}")
#     c1 = model.addConstr(model.addConstr(x1[0] == initial_state[0]), name=f"state1_0")
#     c2 = model.addConstr(model.addConstr(x2[0] == initial_state[1]), name=f"state2_0")
#
#     # 定义目标函数（示例：最小化状态变量和决策变量的平方和）
#     model.setObjective(gp.quicksum(x1[t]**2 + x2[t]**2 for t in range(T)), GRB.MINIMIZE)
#
#
#     # 优化模型
#     model.optimize()
#
#     if model.status == GRB.OPTIMAL:
#         for t in range(T):
#             print(f"x1[{t}] = {x1[t].X}, x2[{t}] = {x2[t].X}, u[{t}] = {u[t].X}, noise[{t}] = {noise[t]}")
#
#     # 更新 noise
#     noise.popleft()
#     noise.append(next(sample_generator))
#
#     # 可选：保存模型
#     model.write('DoubleIntegratorWithNoise.lp')
pass  # 通过将noise定义为决策变量并限制其上下界改变各个时刻的噪声值
# def build_model(T, StateBound, DecisionBound):
#     """
#     构建并返回 Gurobi 模型。
#
#     :param T: 时间步数
#     :param StateBound: 状态变量的边界
#     :param DecisionBound: 决策变量的边界
#     :return: (model, x1, x2, u, noise_vars)
#     """
#     # 创建模型
#     model = gp.Model("ControlOptimization")
#     model.setParam('OutputFlag', 0)
#
#     # 定义变量
#     x1 = model.addVars(T, lb=-StateBound, ub=StateBound, name="x1")  # 状态1
#     x2 = model.addVars(T, lb=-StateBound, ub=StateBound, name="x2")  # 状态2
#     u = model.addVars(T, lb=-DecisionBound, ub=DecisionBound, name="u")  # 输入
#     noise_vars = model.addVars(T - 1, 2, lb=-GRB.INFINITY, name="noise")  # 噪声变量
#
#     # 添加初始状态占位约束
#     model.addConstr(x1[0] == 0, name="initial_x1")
#     model.addConstr(x2[0] == 0, name="initial_x2")
#
#     # 添加状态方程约束
#     for t in range(T - 1):
#         model.addConstr(x1[t + 1] == x1[t] + x2[t] + noise_vars[t, 0], name=f"state1_{t}")
#         model.addConstr(x2[t + 1] == x2[t] + u[t] + noise_vars[t, 1], name=f"state2_{t}")
#
#     # 定义目标函数
#     model.setObjective(gp.quicksum(x1[t] ** 2 + x2[t] ** 2 + u[t] ** 2 for t in range(T)), GRB.MINIMIZE)
#     model.update()
#     return model, x1, x2, u, noise_vars
#
#
# def optimize_control(model, x1, x2, u, noise_vars, x1_0, x2_0, noise):
#     """
#     利用已构建的模型优化决策变量 u。
#
#     :param model: 预先构建的 Gurobi 模型
#     :param x1: 状态变量 x1
#     :param x2: 状态变量 x2
#     :param u: 决策变量 u
#     :param noise_vars: 噪声变量
#     :param x1_0: 初始状态 x1[0]
#     :param x2_0: 初始状态 x2[0]
#     :param noise: 噪声数组，形状为 (T-1, 2)
#     :return: 优化得到的决策变量 u 的值列表
#     """
#     T = len(noise) + 1
#
#     # 更新初始状态约束
#     model.getConstrByName("initial_x1").rhs = x1_0
#     model.getConstrByName("initial_x2").rhs = x2_0
#
#     # 更新噪声值
#     for t in range(T - 1):
#         noise_vars[t, 0].lb = noise[t][0]
#         noise_vars[t, 0].ub = noise[t][0]
#         noise_vars[t, 1].lb = noise[t][1]
#         noise_vars[t, 1].ub = noise[t][1]
#
#     # 优化模型
#     model.optimize()
#
#     if model.status == GRB.OPTIMAL:
#         return [u[t].X for t in range(T)], [x1[t].X for t in range(T)], [x2[t].X for t in range(T)]
#     else:
#         raise RuntimeError("Optimization was not successful. Model status:", model.status)
#
# # 模型参数
# T = 10
# StateBound = 5
# DecisionBound = 5
#
# # 构建模型
# model, x1, x2, u, noise_vars = build_model(T, StateBound, DecisionBound)
#
# # 噪声生成器
# def noise_generator():
#     from random import uniform
#     return [(uniform(-0.1, 0.1), uniform(-0.1, 0.1)) for _ in range(T-1)]
#
# # 初始状态
# x1_0 = 1.0
# x2_0 = -1.0
# noise = noise_generator()
#
# # 优化决策变量
# u_values, x1_values, x2_values = optimize_control(model, x1, x2, u, noise_vars, x1_0, x2_0, noise)
# print("Before x values [", x1_values[0], x2_values[0], "]\n", "Optimized u values:", u_values[0], "\nAfter x values[", x1_values[1], x2_values[1], "]")
#
# # 再次调用（无需重新构建模型）
# x1_0_new = x1_values[1]
# x2_0_new = x2_values[1]
# noise_new = noise_generator()
# u_values, x1_values, x2_values = optimize_control(model, x1, x2, u, noise_vars, x1_0_new, x2_0_new, noise_new)
# print("Before x values [", x1_values[0], x2_values[0], "]\n", "Optimized u values:", u_values[0], "\nAfter x values[", x1_values[1], x2_values[1], "]")
pass  # 通过rhs方法改变约束右边的式子更新各个时刻的噪声值，right hand side，与之对应的由lhs，left hand side
# def build_model_with_rhs(T, StateBound, DecisionBound):
#     """
#     构建并返回 Gurobi 模型。
#
#     :param T: 时间步数
#     :param StateBound: 状态变量的边界
#     :param DecisionBound: 决策变量的边界
#     :return: (model, x1, x2, u, noise_constrs)
#     """
#     # 创建模型
#     model = gp.Model("ControlOptimization")
#     model.setParam('OutputFlag', 0)
#     # 定义变量
#     x1 = model.addVars(T, lb=-StateBound, ub=StateBound, name="x1")  # 状态1
#     x2 = model.addVars(T, lb=-StateBound, ub=StateBound, name="x2")  # 状态2
#     u = model.addVars(T, lb=-DecisionBound, ub=DecisionBound, name="u")  # 输入
#
#     # 添加初始状态占位约束
#     model.addConstr(x1[0] == 0, name="initial_x1")
#     model.addConstr(x2[0] == 0, name="initial_x2")
#
#     # 添加状态方程约束
#     noise_constrs = []
#     for t in range(T - 1):
#         c1 = model.addConstr(x1[t + 1] == x1[t] + x2[t], name=f"state1_{t}")
#         c2 = model.addConstr(x2[t + 1] == x2[t] + u[t], name=f"state2_{t}")
#         noise_constrs.append((c1, c2))
#     model.update()
#     # 定义目标函数
#     model.setObjective(gp.quicksum(x1[t] ** 2 + x2[t] ** 2 + u[t] ** 2 for t in range(T)), GRB.MINIMIZE)
#
#     return model, x1, x2, u, noise_constrs
#
#
# def optimize_control_with_rhs(model, x1, x2, u, noise_constrs, x1_0, x2_0, noise):
#     """
#     利用已构建的模型优化决策变量 u。
#
#     :param model: 预先构建的 Gurobi 模型
#     :param x1: 状态变量 x1
#     :param x2: 状态变量 x2
#     :param u: 决策变量 u
#     :param noise_constrs: 噪声约束
#     :param x1_0: 初始状态 x1[0]
#     :param x2_0: 初始状态 x2[0]
#     :param noise: 噪声数组，形状为 (T-1, 2)
#     :return: 优化得到的决策变量 u 的值列表
#     """
#     T = len(noise) + 1
#
#     # 更新初始状态约束
#     model.getConstrByName("initial_x1").rhs = x1_0
#     model.getConstrByName("initial_x2").rhs = x2_0
#
#     # 更新噪声约束
#     for t in range(T - 1):
#         noise_constrs[t][0].rhs += noise[t][0]  # 更新 x1 方程中的噪声
#         noise_constrs[t][1].rhs += noise[t][1]  # 更新 x2 方程中的噪声
#
#     # 优化模型
#     model.optimize()
#     model.write('DoubleIntegratorWithNoise.lp')
#
#     if model.status == GRB.OPTIMAL:
#         U, X1, X2 = [u[t].X for t in range(T)], [x1[t].X for t in range(T)], [x2[t].X for t in range(T)]
#         # 更新噪声约束
#         for t in range(T - 1):
#             noise_constrs[t][0].rhs -= noise[t][0]  # 更新 x1 方程中的噪声
#             noise_constrs[t][1].rhs -= noise[t][1]  # 更新 x2 方程中的噪声
#         model.write('DoubleIntegratorWithNoise.lp')
#         return U, X1, X2
#     else:
#         raise RuntimeError("Optimization was not successful. Model status:", model.status)
#
#
# # 模型参数
# T = 10
# StateBound = 5
# DecisionBound = 5
#
# # 构建模型
# model, x1, x2, u, noise_vars = build_model_with_rhs(T, StateBound, DecisionBound)
#
# # 噪声生成器
# def noise_generator():
#     from random import uniform
#     return [(uniform(-0.1, 0.1), uniform(-0.1, 0.1)) for _ in range(T-1)]
#
# # 初始状态
# x1_0 = 1.0
# x2_0 = -1.0
# noise = noise_generator()
#
# # 优化决策变量
# u_values, x1_values, x2_values = optimize_control_with_rhs(model, x1, x2, u, noise_vars, x1_0, x2_0, noise)
# print("Before x values [", x1_values[0], x2_values[0], "]\n", "Optimized u values:", u_values[0], "\nAfter x values[", x1_values[1], x2_values[1], "]")
#
# # 再次调用（无需重新构建模型）
# x1_0_new = x1_values[1]
# x2_0_new = x2_values[1]
# noise_new = noise_generator()
# u_values, x1_values, x2_values = optimize_control_with_rhs(model, x1, x2, u, noise_vars, x1_0_new, x2_0_new, noise_new)
# print("Before x values [", x1_values[0], x2_values[0], "]\n", "Optimized u values:", u_values[0], "\nAfter x values[", x1_values[1], x2_values[1], "]")
pass  # 通过将已经实现的各个函数封装成一个类，进一步抽象


class ControlOptimization:
    def __init__(self, T, StateBound, DecisionBound):
        self.T = T
        self.StateBound = StateBound
        self.DecisionBound = DecisionBound
        self.model = None
        self.x1 = None
        self.x2 = None
        self.u = None
        self.X1 = None
        self.X2 = None
        self.U = None
        self.noise_constrs = None
        self.build_model()
        self.X = []

        # 创建外界干扰生成器
        mean = [0.01, 0.01]  # 均值
        std_dev = [0.001, 0.001]  # 标准差
        lower = [0, 0]  # 下界
        upper = [0.02, 0.02]  # 上界
        self.sample_generator = self.truncated_normal_generator(mean, std_dev, lower, upper)
        self.noise = deque([next(self.sample_generator) for _ in range(self.T)])

    def build_model(self):
        # 创建模型
        self.model = gp.Model("ControlOptimization")
        self.model.setParam('OutputFlag', 0)
        # 定义变量
        self.x1 = self.model.addVars(self.T, lb=-self.StateBound, ub=self.StateBound, name="x1")  # 状态1
        self.x2 = self.model.addVars(self.T, lb=-self.StateBound, ub=self.StateBound, name="x2")  # 状态2
        self.u = self.model.addVars(self.T, lb=-self.DecisionBound, ub=self.DecisionBound, name="u")  # 输入

        # 添加初始状态占位约束
        self.model.addConstr(self.x1[0] == 0, name="initial_x1")
        self.model.addConstr(self.x2[0] == 0, name="initial_x2")

        # 添加状态方程约束
        self.noise_constrs = []
        for t in range(self.T - 1):
            c1 = self.model.addConstr(self.x1[t + 1] == self.x1[t] + self.x2[t], name=f"state1_{t}")
            c2 = self.model.addConstr(self.x2[t + 1] == self.x2[t] + self.u[t], name=f"state2_{t}")
            self.noise_constrs.append((c1, c2))
        self.model.update()

        # 定义目标函数
        self.model.setObjective(gp.quicksum(self.x1[t] ** 2 + self.x2[t] ** 2 for t in range(self.T)), GRB.MINIMIZE)

    def optimize_control(self, x1_0, x2_0):
        # 更新初始状态约束
        self.model.getConstrByName("initial_x1").rhs = x1_0
        self.model.getConstrByName("initial_x2").rhs = x2_0

        # 更新噪声约束
        for t in range(self.T - 1):
            self.noise_constrs[t][0].rhs += self.noise[t][0]  # 更新 x1 方程中的噪声
            self.noise_constrs[t][1].rhs += self.noise[t][1]  # 更新 x2 方程中的噪声

        # 优化模型
        self.model.optimize()
        self.model.write('DoubleIntegratorWithNoise.lp')

        if self.model.status == GRB.OPTIMAL:
            self.U = [self.u[t].X for t in range(self.T)]
            self.X1 = [self.x1[t].X for t in range(self.T)]
            self.X2 = [self.x2[t].X for t in range(self.T)]
            self.X.append([self.X1[0], self.X2[0]])

            # 更新噪声约束
            for t in range(self.T - 1):
                self.noise_constrs[t][0].rhs -= self.noise[t][0]  # 更新 x1 方程中的噪声
                self.noise_constrs[t][1].rhs -= self.noise[t][1]  # 更新 x2 方程中的噪声

            self.noise.popleft()
            self.noise.append(next(self.sample_generator))
            self.model.write('DoubleIntegratorWithNoise.lp')

        else:
            raise RuntimeError("Optimization was not successful. Model status:", self.model.status)

    def truncated_normal_generator(self, mean, std_dev, lower, upper):
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


# 参数设置
T = 10
StateBound = GRB.INFINITY
DecisionBound = 5
x1_0 = 200.0
x2_0 = -200.0
optimizer = ControlOptimization(T, StateBound, DecisionBound)
# 优化决策变量
optimizer.optimize_control(x1_0, x2_0)
print("Before x values [", optimizer.X1[0], optimizer.X2[0], "]\n", "Optimized u values:", optimizer.U[0],
      "\nAfter x values[", optimizer.X1[1], optimizer.X2[1], "]")

# while abs(optimizer.X1[1])>0.005 or abs(optimizer.X2[1])>0.005:
for _ in range(1000):
    # 再次调用（无需重新构建模型）
    optimizer.optimize_control(optimizer.X1[1], optimizer.X2[1])
    print("Optimized u values:", optimizer.U[0], "\nAfter x values[", optimizer.X1[1], optimizer.X2[1], "]")

# 绘制 optimizer.X1 和 optimizer.X2 的折线图
plt.figure(figsize=(10, 6))
plt.plot([row[0] for row in optimizer.X], label='X1')
plt.plot([row[1] for row in optimizer.X], label='X2')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('States changes of X1 and X2')
plt.legend()
plt.show()
