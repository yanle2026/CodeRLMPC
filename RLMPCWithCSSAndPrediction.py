import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import truncnorm
from PredictorTrainerLSTMPyTorch import LSTMModel
import joblib, torch

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


class RLMPCWithoutRCSButPrediction:
    def __init__(self, T=10, StateBound=GRB.INFINITY, DecisionBound=5, NoiseGenerator=noise(amplitude=1, std_dev=0.1)):
        # 建模相关变量
        self.T = T
        self.StateBound = StateBound
        self.DecisionBound = DecisionBound
        self.model = None
        self.x1 = None
        self.x2 = None
        self.u = None
        self.lamb = None
        self.noise_constrs = None
        # 当前迭代的整个轨迹
        self.X = []
        self.U = []
        self.TerminalQ = deque()
        self.Points = 0 # 当前轨迹中状态点的个数
        # 安全集
        self.SSX = []
        self.SSU = []
        self.SSQ = []
        self.SSPointsNum = None
        self.SSXNew = []
        self.SSUNew = []
        self.SSQNew = []
        # 初始化是按顺序来的，函数内不要操作还未声明的变量
        self.build_model()

        # 外界噪声生成器
        self.sample_generator = NoiseGenerator
        self.history_noise = deque()    # 历史 T 个历史外界噪声
        self.predicted_noise = []  # 预测的未来 T 个外界噪声

        # 重新加载模型实例
        self.loaded_model = LSTMModel(input_size=2, hidden_size=128, output_size=2, num_layers=3)
        self.loaded_model.load_state_dict(torch.load('predictor_model.pth'))  # 加载模型权重
        self.loaded_model.eval()  # 切换到评估模式
        self.scaler = joblib.load('scaler.pkl')

    def build_model(self):
        # 加载安全集
        self.load_statics()
        # 创建模型
        self.model = gp.Model("ControlOptimization")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('FeasibilityTol', 1e-5)  # 放宽可行性容忍度
        self.model.setParam('NumericFocus', 3)  # 增强对数值稳定性的关注
        self.model.setParam('ScaleFlag', 1)
        self.model.update()
        # 定义变量
        self.x1 = self.model.addVars(self.T+1, lb=-self.StateBound, ub=self.StateBound, vtype=GRB.CONTINUOUS, name="x1")  # 状态1
        self.x2 = self.model.addVars(self.T+1, lb=-self.StateBound, ub=self.StateBound, vtype=GRB.CONTINUOUS, name="x2")  # 状态2
        self.u = self.model.addVars(self.T, lb=-self.DecisionBound, ub=self.DecisionBound, vtype=GRB.CONTINUOUS, name="u")  # 输入
        self.lamb = self.model.addVars(self.SSPointsNum, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="lamb")# 凸组合系数
        # 添加初始状态占位约束
        self.model.addConstr(self.x1[0] == 0, name="initial_x1")
        self.model.addConstr(self.x2[0] == 0, name="initial_x2")
        # 添加状态方程约束
        self.noise_constrs = []
        for t in range(self.T):
            c1 = self.model.addConstr(self.x1[t + 1] == self.x1[t] + self.x2[t], name=f"state1_{t}")
            c2 = self.model.addConstr(self.x2[t + 1] == self.x2[t] + self.u[t], name=f"state2_{t}")
            self.noise_constrs.append((c1, c2))
        # 添加终端状态约束
        self.model.addConstr(
            self.x1[self.T] == gp.quicksum(self.lamb[i] * self.SSX[i][0] for i in range(self.SSPointsNum)),
            name=f"convex_hull_dim_1"
        )
        self.model.addConstr(
            self.x2[self.T] == gp.quicksum(self.lamb[i] * self.SSX[i][1] for i in range(self.SSPointsNum)),
            name=f"convex_hull_dim_2"
        )
        self.model.addConstr(
            gp.quicksum(self.lamb[i] for i in range(self.SSPointsNum)) == 1,
            name="convex_combination_sum"
        )# 添加 lamb 的总和为1的约束
        self.model.update()
        # 定义目标函数
        self.model.setObjective(gp.quicksum(self.x1[t] ** 2 + self.x2[t] ** 2 + self.u[t] ** 2 for t in range(self.T))
                                + gp.quicksum(self.lamb[i] * self.SSQ[i] for i in range(self.SSPointsNum)),
                                GRB.MINIMIZE)

    def optimize_control(self, x1_0, x2_0):
        times = 0
        x1 = x1_0
        x2 = x2_0
        self.refresh_terminal_constrs()
        # 给任务启动时的空历史干扰序列补充零值或其他假设值（可以根据具体场景调整）
        if len(self.history_noise) < self.T:
            padding = np.zeros((self.T - len(self.history_noise), 2))
            for _ in range(len(padding)):
                self.history_noise.appendleft(padding[_])
        # 根据历史 T 个历史外界噪声，预测未来 T 个外界噪声，保存在 self.predicted_noise，并用于 MPC 算法
        self.predict_noise()

        while True:
            # 更新初始状态约束
            self.model.getConstrByName("initial_x1").rhs = x1
            self.model.getConstrByName("initial_x2").rhs = x2

            # 通过预测所得外界干扰进行决策
            # 更新噪声约束
            for t in range(self.T):
                self.noise_constrs[t][0].rhs += self.predicted_noise[t][0]  # 更新 x1 方程中的噪声
                self.noise_constrs[t][1].rhs += self.predicted_noise[t][1]  # 更新 x2 方程中的噪声

            # 优化模型
            self.model.optimize()
            self.model.write('DoubleIntegratorWithNoise.lp')

            if self.model.status == GRB.OPTIMAL:
                self.U.append(self.u[0].X)
                self.X.append([self.x1[0].X, self.x2[0].X])
                self.Points += 1
                x1 = self.x1[1].X+self.history_noise[-1][0]
                x2 = self.x2[1].X+self.history_noise[-1][1]
                # 判断系统状态是不是已经进入了不变集O；且因为加入了外界干扰，判断算法停止的条件应该放松
                if abs(x1) < 1.1 + 1e-5 and abs(x2) < 1.1 + 1e-5:
                    self.X.append([self.x1[1].X, self.x2[1].X])
                    break
                # 更新噪声约束
                for t in range(self.T - 1):
                    self.noise_constrs[t][0].rhs -= self.predicted_noise[t][0]  # 更新 x1 方程中的噪声
                    self.noise_constrs[t][1].rhs -= self.predicted_noise[t][1]  # 更新 x2 方程中的噪声
                self.history_noise.append(next(self.sample_generator))
                self.history_noise.popleft()
                self.model.write('DoubleIntegratorWithNoise.lp')
            else:
                self.model.write('ErrorModel.lp')
                print("Retrying optimization...")
                self.model.optimize()
                if self.model.status == GRB.OPTIMAL:
                    print("Model is actually feasible.")
                print("Model is infeasible. Computing IIS...")
                self.model.computeIIS()  # 计算不可行的约束集
                self.model.write("infeasible_model.ilp")  # 保存不可行的子集
                print("Infeasible model saved to infeasible_model.ilp")
                raise RuntimeError("Optimization was not successful. Model status:", self.model.status)
        # 将当前迭代的轨迹X、U、Q添加进SSNew（按迭代保存用）和SS（整个存放计算用）
        self.SSXNew.append(np.array(self.X))
        self.SSUNew.append(np.array(self.U))
        self.SSX = np.concatenate(self.SSXNew)
        self.SSU = np.concatenate(self.SSUNew)
        self.SSQNew_append()

    def predict_noise(self):
        # 利用 self.history_noise 中的数据预测未来 T 个外界噪声，并存储在 self.predicted_noise
        history_noise = self.history_noise
        self.predicted_noise.clear()

        for i in range(self.T):
            history_scaled = self.scaler.transform(self.history_noise)
            # 转换为 PyTorch 张量，并增加 batch 维度
            history_tensor = torch.tensor(history_scaled, dtype=torch.float32).unsqueeze(0)
            # 使用加载的模型进行预测
            with torch.no_grad():
                predicted_scaled = self.loaded_model(history_tensor).squeeze(0).numpy()
            # 反归一化预测值
            predicted_noise = self.scaler.inverse_transform(predicted_scaled.reshape(1, -1))
            # print("Predicted Noise:", predicted_noise.squeeze(0))
            self.predicted_noise.append(predicted_noise.squeeze(0))
            history_noise.popleft()
            history_noise.append(predicted_noise.squeeze(0))

    def refresh_terminal_constrs(self):
        # 更新安全集信息
        self.SSPointsNum = len(self.SSX)
        # 更新凸组合系数变量
        for var in self.lamb.values():
            self.model.remove(var)# 删除旧变量
        self.model.update()
        self.lamb = self.model.addVars(self.SSPointsNum, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="lamb")  # 添加新凸组合系数
        # 更新终端状态约束
        # 删除终端状态约束
        self.model.remove(self.model.getConstrByName("convex_hull_dim_1"))
        self.model.remove(self.model.getConstrByName("convex_hull_dim_2"))
        self.model.remove(self.model.getConstrByName("convex_combination_sum"))
        self.model.addConstr(
            self.x1[self.T] == gp.quicksum(self.lamb[i] * self.SSX[i][0] for i in range(self.SSPointsNum)),
            name=f"convex_hull_dim_1"
        )
        self.model.addConstr(
            self.x2[self.T] == gp.quicksum(self.lamb[i] * self.SSX[i][1] for i in range(self.SSPointsNum)),
            name=f"convex_hull_dim_2"
        )
        self.model.addConstr(
            gp.quicksum(self.lamb[i] for i in range(self.SSPointsNum)) == 1,
            name="convex_combination_sum"
        )# 添加 lamb 的总和为1的约束
        self.model.update()
        # 定义目标函数
        self.model.setObjective(gp.quicksum(self.x1[t] ** 2 + self.x2[t] ** 2 + self.u[t] ** 2 for t in range(self.T))
                                + gp.quicksum(self.lamb[i] * self.SSQ[i] for i in range(self.SSPointsNum)),
                                GRB.MINIMIZE)

    def SSQNew_append(self):
        self.TerminalQ.append(0)
        self.X = [sum(x**2 for x in row) for row in self.X]
        self.U = [row**2 for row in self.U]
        self.X.pop()
        while self.Points:
            self.TerminalQ.appendleft(self.X[-1] + self.U[-1] + self.TerminalQ[0])
            self.U.pop()
            self.X.pop()
            self.Points -= 1
        self.SSQNew.append(np.array(self.TerminalQ))
        self.SSQ = np.concatenate(self.SSQNew)
        self.TerminalQ.clear()

    def save_statics(self):
        # 将训练所得的历史数据保存为文件
        # 对于大型数据集和数值计算，NumPy数组在内存占用和运行速度方面通常优于Python列表。
        # 但是，如果数据类型不固定或者需要进行频繁的插入和删除操作，Python列表可能更合适。
        np.savez('SSX.npz', *self.SSXNew)
        np.savez('SSU.npz', *self.SSUNew)
        np.savez('SSQ.npz', *self.SSQNew)

    def load_statics(self):
        # 加载可行轨迹
        loaded_arrays = np.load('FeasibleX.npz')
        self.SSXNew = [loaded_arrays[key] for key in loaded_arrays.files]
        self.SSX = np.concatenate(self.SSXNew)
        loaded_arrays = np.load('FeasibleU.npz')
        self.SSUNew = [loaded_arrays[key] for key in loaded_arrays.files]
        self.SSU = np.concatenate(self.SSUNew)
        loaded_arrays = np.load('FeasibleQ.npz')
        self.SSQNew = [loaded_arrays[key] for key in loaded_arrays.files]
        self.SSQ = np.concatenate(self.SSQNew)
        self.SSPointsNum = len(self.SSX)

    def plot_trajectory(self):
        # 绘制 optimizer.X1 和 optimizer.X2 的折线图
        plt.figure()
        x1 = [row[0] for row in self.SSXNew[-1]]
        x2 = [row[1] for row in self.SSXNew[-1]]
        plt.plot(x1, x2, 'm')
        plt.scatter(x1, x2, s=5, c='blue', marker='o')
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

    def plot_cost_trajectory(self):
        plt.figure()
        plt.plot([i[0]/100 for i in self.SSQNew[5:]])
        plt.show()


if __name__ == "__main__":
    # 参数设置
    T = 10
    StateBound = GRB.INFINITY
    DecisionBound = GRB.INFINITY
    x1_0 = 200.0
    x2_0 = -200.0

    NoiseGenerator = noise(amplitude=1, std_dev=0.1)
    optimizer = RLMPCWithoutRCSButPrediction(T, StateBound, DecisionBound, NoiseGenerator)
    Cost = 0

    for _ in range(20):
        optimizer.optimize_control(x1_0, x2_0)
        Cost = optimizer.SSQNew[-1][0]
        print(f"第{_+1}次学习，成本为{Cost}")

    # optimizer.save_statics()
    optimizer.plot_trajectory()
    optimizer.plot_cost_trajectory()
