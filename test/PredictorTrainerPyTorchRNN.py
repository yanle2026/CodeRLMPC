import torch
import torch.nn as nn
import numpy as np


# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取RNN的最后一个输出
        return out


# 生成一些示例数据
def generate_synthetic_data(n_samples=1000, n_timesteps=10):
    # 假设外界噪声遵循截断正态分布
    noise_data = np.random.normal(0, 1, size=(n_samples, n_timesteps, 2))
    return noise_data


# 数据预处理
def preprocess_data(noise_data):
    X = []
    y = []
    for i in range(len(noise_data) - 1):
        X.append(noise_data[i])  # 前10个时刻的噪声作为输入
        y.append(noise_data[i + 1, -1, :])  # 下一个时刻的噪声作为输出
    return np.array(X), np.array(y)


# 设置超参数
input_size = 2  # 外界噪声的维度（二维）
hidden_size = 64  # RNN隐藏层大小
output_size = 2  # 预测的噪声维度
n_epochs = 100
batch_size = 32
learning_rate = 0.001

# 生成数据并预处理
data = generate_synthetic_data()
X_train, y_train = preprocess_data(data)

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# 初始化模型、损失函数和优化器
model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}")


# 迭代预测未来时刻的噪声
def iterative_prediction(model, initial_input, num_predictions=10):
    model.eval()
    predictions = []
    input_seq = initial_input

    for _ in range(num_predictions):
        with torch.no_grad():
            predicted = model(input_seq)
            predictions.append(predicted.numpy())

            # 更新输入序列，加入预测值
            input_seq = torch.cat((input_seq[:, 1:, :], predicted.unsqueeze(1)), dim=1)

    return np.array(predictions)


# 用最后10个时刻的数据进行迭代预测
initial_input = X_train[-1:, :, :]
predictions = iterative_prediction(model, initial_input, num_predictions=10)

# 输出预测结果
print("Predicted noise values for the next 10 timesteps:")
print(predictions)
