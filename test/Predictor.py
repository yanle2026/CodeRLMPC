import torch
import torch.nn as nn
import numpy as np

# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)  # 初始化隐藏状态
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        out = self.sigmoid(out)  # 将输出限制在(0,1)之间
        return out

# 定义边界的缩放函数
def rescale_output(output, low_bound, high_bound):
    return low_bound + output * (high_bound - low_bound)

# 模型参数
input_size = 2    # 输入特征的维度
hidden_size = 64  # RNN隐藏层的维度
output_size = 2   # 输出的维度（假设输出两个数值）
sequence_length = 10  # 时间步数
num_epochs = 50   # 训练的轮数
batch_size = 32   # 每个批次的大小
learning_rate = 0.001

# 创建模型
model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 生成随机训练数据 (样本数量, 时间步数, 特征数量)
history_data = torch.tensor(np.random.rand(1000, sequence_length, input_size), dtype=torch.float32)
target_data = torch.tensor(np.random.rand(1000, output_size), dtype=torch.float32)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(history_data), batch_size):
        inputs = history_data[i:i + batch_size]
        targets = target_data[i:i + batch_size]
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测新数据
model.eval()
new_data = torch.tensor(np.random.rand(1, sequence_length, input_size), dtype=torch.float32)
with torch.no_grad():
    outputs = model(new_data)  

print(outputs)
print(target_data[-1])