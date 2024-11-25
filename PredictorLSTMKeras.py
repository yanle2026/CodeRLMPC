# import numpy as np
# import torch
# from sklearn.preprocessing import MinMaxScaler
# from torch.utils.data import DataLoader, Dataset
# from scipy.stats import truncnorm, norm
#
# # 截断正态分布生成函数
# def truncated_normal_generator(mean, std_dev, lower, upper, size):
#     a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
#     dist = truncnorm(a, b, loc=mean, scale=std_dev)
#     return dist.rvs(size)
#
# # 生成二维噪声数据
# def generate_data(mean, std_dev, lower, upper, timesteps, num_samples):
#     x_noise = truncated_normal_generator(mean[0], std_dev[0], lower[0], upper[0], num_samples)
#     y_noise = truncated_normal_generator(mean[1], std_dev[1], lower[1], upper[1], num_samples)
#     data = np.array([x_noise, y_noise]).T  # 形状: (num_samples, 2)
#     return data
#
# # 参数设置
# mean = [0, 0]
# std_dev = [0.1, 0.1]
# lower = [-1, -1]
# upper = [1, 1]
# timesteps = 10  # 每个样本的时间步
# num_samples = 1000  # 总样本数
#
# # 生成数据
# data = generate_data(mean, std_dev, lower, upper, timesteps, num_samples)
#
# # 数据归一化
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data)
# # 创建时间序列数据
# def create_time_series(data, timesteps):
#     X, y = [], []
#     for i in range(len(data) - timesteps):
#         X.append(data[i:i+timesteps])
#         y.append(data[i+timesteps])
#     return np.array(X), np.array(y)
#
# # 构建时间序列
# X, y = create_time_series(scaled_data, timesteps)
#
# # 划分训练集和测试集
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]
#
# # 转换为 PyTorch 张量
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#
# # 创建 PyTorch 数据集
# class TimeSeriesDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
# # 构造 DataLoader
# train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
# test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# import torch.nn as nn
#
# # 定义 LSTM 模型
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)  # 输出形状: (batch_size, seq_len, hidden_size)
#         out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
#         return out
#
# # 初始化模型
# input_size = 2  # 噪声的二维特征
# hidden_size = 64
# output_size = 2
# num_layers = 2
#
# model = LSTMModel(input_size, hidden_size, output_size, num_layers)
# import torch.optim as optim
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练函数
# def train_model(model, train_loader, criterion, optimizer, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for X_batch, y_batch in train_loader:
#             # 前向传播
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
#
# # 测试函数
# def evaluate_model(model, test_loader):
#     model.eval()
#     predictions, actuals = [], []
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             outputs = model(X_batch)
#             predictions.append(outputs.numpy())
#             actuals.append(y_batch.numpy())
#     return np.vstack(predictions), np.vstack(actuals)
# # 训练模型
# num_epochs = 10
# train_model(model, train_loader, criterion, optimizer, num_epochs)
#
# # 在测试集上预测
# y_pred, y_true = evaluate_model(model, test_loader)
# print(f"测试集 loss 的值为{np.mean((y_true - y_pred) ** 2)}")
# # 反归一化预测值和真实值
# y_pred_rescaled = scaler.inverse_transform(y_pred)
# y_true_rescaled = scaler.inverse_transform(y_true)
#
# # 可视化结果
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 6))
# plt.plot(y_true_rescaled[:100, 0], label='True X (Noise)')
# plt.plot(y_pred_rescaled[:100, 0], label='Predicted X (Noise)')
# plt.plot(y_true_rescaled[:100, 1], label='True Y (Noise)')
# plt.plot(y_pred_rescaled[:100, 1], label='Predicted Y (Noise)')
# plt.legend()
# plt.title("LSTM Noise Prediction (PyTorch)")
# plt.show()
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from scipy.stats import truncnorm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class LSTMNoisePredictor:
    """
    LSTMNoisePredictor 是一个基于LSTM模型的预测器，用于预测时间序列数据中的噪声信号。

    参数:
    - mean: 噪声信号的均值。
    - std_dev: 噪声信号的标准差。
    - lower: 噪声信号的下界。
    - upper: 噪声信号的上界。
    - timesteps: 时间序列中的时间步长数。
    - num_samples: 生成数据的样本数量。
    - input_size: LSTM模型的输入特征数量。
    - hidden_size: LSTM模型隐藏层的神经元数量。
    - output_size: LSTM模型的输出特征数量。
    - num_layers: LSTM模型中的层数。

    方法:
    - _truncated_normal_generator: 生成截断正态分布的数据。
    - _generate_data: 生成时间序列数据。
    - _create_time_series: 将数据转换为时间序列格式。
    - _prepare_data: 预处理数据，包括归一化和划分训练/测试集。
    - train: 训练LSTM模型。
    - evaluate: 评估LSTM模型在测试数据上的性能。
    - plot_results: 绘制真实值和预测值的对比图。
    """
    def __init__(self, mean, std_dev, lower, upper, timesteps, num_samples, input_size, hidden_size, output_size, num_layers):
        self.mean = mean
        self.std_dev = std_dev
        self.lower = lower
        self.upper = upper
        self.timesteps = timesteps
        self.num_samples = num_samples

        # 数据预处理
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_loader, self.test_loader = self._prepare_data()

        # 模型初始化
        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _truncated_normal_generator(self, mean, std_dev, lower, upper):
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

        while True:
            # 通过分布生成一个新的样本值
            yield np.array([dist_x.rvs(), dist_y.rvs()])

    def _generate_data(self):
        sample_generator = self._truncated_normal_generator(self.mean, self.std_dev, self.lower, self.upper)
        samples = np.array([next(sample_generator) for _ in range(self.num_samples)])
        return samples

    def _create_time_series(self, data):
        X, y = [], []
        for i in range(len(data) - self.timesteps):
            X.append(data[i:i+self.timesteps])
            y.append(data[i+self.timesteps])
        return np.array(X), np.array(y)

    def _prepare_data(self):
        data = self._generate_data()
        scaled_data = self.scaler.fit_transform(data)

        X, y = self._create_time_series(scaled_data)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
        test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(self.train_loader):.4f}")

    def evaluate(self):
        self.model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                outputs = self.model(X_batch)
                predictions.append(outputs.numpy())
                actuals.append(y_batch.numpy())
        y_pred = np.vstack(predictions)
        y_true = np.vstack(actuals)
        loss = np.mean((y_true - y_pred) ** 2)
        print(f"Test Loss: {loss:.4f}")

        y_pred_rescaled = self.scaler.inverse_transform(y_pred)
        y_true_rescaled = self.scaler.inverse_transform(y_true)
        return y_pred_rescaled, y_true_rescaled

    def plot_results(self, y_pred_rescaled, y_true_rescaled, num_points=100):
        plt.figure(figsize=(12, 6))
        plt.plot(y_true_rescaled[:num_points, 0], label='True X (Noise)')
        plt.plot(y_pred_rescaled[:num_points, 0], label='Predicted X (Noise)')
        plt.plot(y_true_rescaled[:num_points, 1], label='True Y (Noise)')
        plt.plot(y_pred_rescaled[:num_points, 1], label='Predicted Y (Noise)')
        plt.legend()
        plt.title("LSTM Noise Prediction (PyTorch)")
        plt.show()

    def get_next_noise_prediction(self, latest_data_point):
        """
        获取下一个时刻的真实噪声值和预测噪声值。

        参数:
        - latest_data_point: 最新的数据点，用于预测下一个时间点的噪声值。

        返回:
        - true_next_noise: 下一个时刻的真实噪声值。
        - predicted_next_noise: 下一个时刻的预测噪声值。
        """
        # 预处理最新的数据点以匹配模型输入格式
        latest_data_point_scaled = self.scaler.transform(latest_data_point.reshape(1, -1))
        latest_data_point_tensor = torch.tensor(latest_data_point_scaled, dtype=torch.float32).unsqueeze(0)  # 增加批次维度

        # 使用模型预测下一个时间点的噪声值
        self.model.eval()
        with torch.no_grad():
            predicted_next_noise_tensor = self.model(latest_data_point_tensor)
            predicted_next_noise = predicted_next_noise_tensor.numpy().flatten()

        # 获取下一个时刻的真实噪声值
        # 注意：这里假设latest_data_point已经是真实噪声值，因此不需要额外的转换
        true_next_noise = latest_data_point[-1]  # 假设latest_data_point的最后一个元素是下一个时间点的真实噪声值

        # 反归一化预测值
        predicted_next_noise_rescaled = self.scaler.inverse_transform(predicted_next_noise.reshape(1, -1)).flatten()

        return true_next_noise, predicted_next_noise_rescaled


# 示例使用：
# 假设我们有一个LSTMNoisePredictor实例叫做predictor，并且我们有最新的数据点latest_data_point
# true_noise, pred_noise = predictor.get_next_noise_prediction(latest_data_point)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # 输出形状: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 示例使用
predictor = LSTMNoisePredictor(mean=[0, 0], std_dev=[0.1, 0.1], lower=[-1, -1], upper=[1, 1],
                                timesteps=10, num_samples=1000,
                                input_size=2, hidden_size=64, output_size=2, num_layers=2)

predictor.train(num_epochs=10)
y_pred_rescaled, y_true_rescaled = predictor.evaluate()
predictor.plot_results(y_pred_rescaled, y_true_rescaled)
predictor._
predictor.model()