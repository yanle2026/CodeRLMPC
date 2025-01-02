import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from SineNormalNoiseGenerator import noise
from PredictorTrainerLSTMPyTorch import LSTMNoisePredictor, LSTMModel
import joblib, torch


class ControlOptimizationWithoutPrediction:
    def __init__(self, T, StateBound, DecisionBound, NoiseGenerator):
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

        # 外界噪声生成器
        self.sample_generator = NoiseGenerator
        self.noise = deque([next(self.sample_generator) for _ in range(self.T)])

        # 重新加载模型实例
        self.loaded_model = LSTMModel(input_size=2, hidden_size=128, output_size=2, num_layers=3)
        self.loaded_model.load_state_dict(torch.load('predictor_model.pth'))  # 加载模型权重
        self.loaded_model.eval()  # 切换到评估模式

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

    def plot_trajectory(self):
        # 绘制 optimizer.X1 和 optimizer.X2 的折线图
        plt.figure(figsize=(10, 6))
        plt.plot([row[0] for row in self.X], label='X1')
        plt.plot([row[1] for row in self.X], label='X2')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('States changes of X1 and X2')
        plt.legend()
        plt.show()


class ControlOptimizationWithPrediction:
    def __init__(self, T, StateBound, DecisionBound, NoiseGenerator):
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

        # 添加状态方程约束，不带噪声值
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

        if len(self.history_noise) < self.T:
            # 补充零值或其他假设值（可以根据具体场景调整）
            padding = np.zeros((self.T - len(self.history_noise), 2))
            for _ in range(len(padding)):
                self.history_noise.appendleft(padding[_])

        # 根据历史 T 个历史外界噪声，预测未来 T 个外界噪声，保存在 self.predicted_noise，并用于 MPC 算法
        self.predict_noise()

        # 更新噪声约束
        for t in range(self.T - 1):
            self.noise_constrs[t][0].rhs += self.predicted_noise[t][0]  # 更新 x1 方程中的噪声
            self.noise_constrs[t][1].rhs += self.predicted_noise[t][1]  # 更新 x2 方程中的噪声

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
                self.noise_constrs[t][0].rhs -= self.predicted_noise[t][0]  # 更新 x1 方程中的噪声
                self.noise_constrs[t][1].rhs -= self.predicted_noise[t][1]  # 更新 x2 方程中的噪声

            self.history_noise.append(next(self.sample_generator))
            self.history_noise.popleft()
            self.model.write('DoubleIntegratorWithNoise.lp')

        else:
            raise RuntimeError("Optimization was not successful. Model status:", self.model.status)

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

    def plot_trajectory(self):
        # 绘制 optimizer.X1 和 optimizer.X2 的折线图
        plt.figure(figsize=(10, 6))
        plt.plot([row[0] for row in self.X], label='X1')
        plt.plot([row[1] for row in self.X], label='X2')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title('States changes of X1 and X2')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # 参数设置
    T = 10
    StateBound = GRB.INFINITY
    DecisionBound = 5
    x1_0 = 200.0
    x2_0 = -200.0
    optimizer = ControlOptimizationWithoutPrediction(T, StateBound, DecisionBound, NoiseGenerator=noise(amplitude=5, std_dev=1))
    # 优化决策变量
    optimizer.optimize_control(x1_0, x2_0)
    print("Before x values [", optimizer.X1[0], optimizer.X2[0], "]\n", "Optimized u values:", optimizer.U[0],
          "\nAfter x values[", optimizer.X1[1], optimizer.X2[1], "]")

    # while abs(optimizer.X1[1])>0.005 or abs(optimizer.X2[1])>0.005:
    for _ in range(1000):
        # 再次调用（无需重新构建模型）
        optimizer.optimize_control(optimizer.X1[1], optimizer.X2[1])
        print("Optimized u values:", optimizer.U[0], "\nAfter x values[", optimizer.X1[1], optimizer.X2[1], "]")

    optimizer.plot_trajectory()
