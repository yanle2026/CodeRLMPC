from collections import deque
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from SineNormalNoiseGenerator import noise
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib


class LSTMNoisePredictor:
    """
    LSTMNoisePredictor 是一个基于LSTM模型的预测器，用于预测时间序列数据中的噪声信号。

    参数:
    - noise_generator: 噪声生成器
    - timesteps: 时间序列中的时间步长数。
    - num_samples: 生成数据的样本数量。
    - input_size: LSTM模型的输入特征数量。
    - hidden_size: LSTM模型隐藏层的神经元数量。
    - output_size: LSTM模型的输出特征数量。
    - num_layers: LSTM模型中的层数。

    方法:
    - truncated_normal_generator: 生成截断正态分布的数据。
    - _generate_data: 生成时间序列数据。
    - _create_time_series: 将数据转换为时间序列格式。
    - _prepare_data: 预处理数据，包括归一化和划分训练/测试集。
    - train: 训练LSTM模型。
    - evaluate: 评估LSTM模型在测试数据上的性能。
    - plot_results: 绘制真实值和预测值的对比图。
    """

    def __init__(self, noise_generator, timesteps, num_samples, input_size, hidden_size, output_size, num_layers):
        self.noise_generator = noise_generator
        self.timesteps = timesteps
        self.num_samples = num_samples

        # 数据预处理
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_loader, self.test_loader = self._prepare_data()

        # 模型初始化
        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def _generate_data(self):
        samples = np.array([next(self.noise_generator) for _ in range(self.num_samples)])
        return samples

    def _create_time_series(self, data):
        X, y = [], []
        for i in range(len(data) - self.timesteps):
            X.append(data[i:i + self.timesteps])
            y.append(data[i + self.timesteps])
        return np.array(X), np.array(y)

    def _prepare_data(self):
        data = self._generate_data()
        scaled_data = self.scaler.fit_transform(data)
        # 保存 scaler
        joblib.dump(self.scaler, 'scaler.pkl')

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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(self.train_loader):.5f}")

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

    @staticmethod
    def plot_results(y_pred_rescaled, y_true_rescaled, num_points=100):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))  # 创建一个包含两个子图的窗口

        # 第一个子图显示X轴的数据
        ax1.plot(y_true_rescaled[:num_points, 0], label='True X (Noise)')
        ax1.plot(y_pred_rescaled[:num_points, 0], label='Predicted X (Noise)')
        ax1.plot(y_true_rescaled[:num_points, 0]-y_pred_rescaled[:num_points, 0], label='Bias X (Noise)')
        ax1.set_title('X-axis Noise Prediction')
        ax1.legend()

        # 第二个子图显示Y轴的数据
        ax2.plot(y_true_rescaled[:num_points, 1], label='True Y (Noise)')
        ax2.plot(y_pred_rescaled[:num_points, 1], label='Predicted Y (Noise)')
        ax2.plot(y_true_rescaled[:num_points, 1]-y_pred_rescaled[:num_points, 1], label='Bias X (Noise)')
        ax2.set_title('Y-axis Noise Prediction')
        ax2.legend()

        plt.tight_layout()  # 调整子图间距
        plt.show()

    def predict_next_noise(self, history):
        """
        根据历史噪声预测下一个时刻的噪声值
        :param history: 最近 timesteps 个噪声值，形状为 (timesteps, input_size)
        :return: 预测的下一个噪声值，形状为 (output_size,)
        """
        self.model.eval()  # 切换到评估模式
        if len(history) != self.timesteps:
            raise ValueError(f"历史噪声长度必须为 {self.timesteps}，但收到 {len(history)}")

        # 将历史数据归一化，并转换为 PyTorch 张量
        history_scaled = self.scaler.transform(history)  # 归一化
        history_tensor = torch.tensor(history_scaled, dtype=torch.float32).unsqueeze(0)  # 添加 batch 维度

        with torch.no_grad():
            predicted_scaled = self.model(history_tensor).squeeze(0).numpy()  # 模型输出

        # 反归一化预测值
        predicted_noise = self.scaler.inverse_transform(predicted_scaled.reshape(1, -1))
        return predicted_noise.squeeze(0)


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


if __name__ == '__main__':
    # 噪声生成器
    noise_generator = noise()
    # 噪声预测器
    predictor = LSTMNoisePredictor(noise_generator=noise_generator, timesteps=10, num_samples=10000,
                                   input_size=2, hidden_size=128, output_size=2, num_layers=3)
    predictor.train(num_epochs=5)
    y_pred_rescaled, y_true_rescaled = predictor.evaluate()
    LSTMNoisePredictor.plot_results(y_pred_rescaled, y_true_rescaled)
    # 保存模型权重
    torch.save(predictor.model.state_dict(), 'predictor_model.pth')
    # 保存优化器状态
    torch.save(predictor.optimizer.state_dict(), 'optimizer_state.pth')

