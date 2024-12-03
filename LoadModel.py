from collections import deque
import joblib, torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from PredictorTrainerLSTMPyTorch import LSTMNoisePredictor, LSTMModel
# 重新创建模型实例
loaded_model = LSTMModel(input_size=2, hidden_size=128, output_size=2, num_layers=3)
loaded_model.load_state_dict(torch.load('predictor_model.pth'))  # 加载模型权重
loaded_model.eval()  # 切换到评估模式
# # 如果需要恢复优化器（可选）
# optimizer = optim.Adam(loaded_model.parameters(), lr=0.001)
# optimizer.load_state_dict(torch.load('optimizer_state.pth'))
# 噪声生成器
noise_generator = LSTMNoisePredictor.truncated_normal_generator(mean=[0, 0], std_dev=[0.1, 0.1], lower=[-1, -1],
                                                                upper=[1, 1])
pre = []
true = []
noise = np.array([next(noise_generator) for _ in range(10)])

# 数据归一化（确保使用相同的 scaler）
# 加载 scaler
scaler = joblib.load('scaler.pkl')

for i in range(100):
    history_scaled = scaler.transform(noise)
    # 转换为 PyTorch 张量，并增加 batch 维度
    history_tensor = torch.tensor(history_scaled, dtype=torch.float32).unsqueeze(0)
    # 使用加载的模型进行预测
    with torch.no_grad():
        predicted_scaled = loaded_model(history_tensor).squeeze(0).numpy()
    # 反归一化预测值
    predicted_noise = scaler.inverse_transform(predicted_scaled.reshape(1, -1))
    print("Predicted Noise:", predicted_noise.squeeze(0))
    pre.append(predicted_noise.squeeze(0))
    true.append(next(noise_generator))
    print("True Noise:", true[-1])
    noise = deque(noise)
    noise.popleft()
    noise.append(true[-1])
    noise = np.array(noise)

LSTMNoisePredictor.plot_results(np.array(pre), np.array(true), num_points=100)
# 生成样本并转换为数组
bias = np.array(pre) - np.array(true)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))  # 创建一个包含两个子图的窗口

# 第一个子图显示X轴的数据
ax1.plot(bias[:, 0])
ax1.set_title('Bias X (Noise)')
ax1.legend()

# 第二个子图显示Y轴的数据
ax2.plot(bias[:, 1])
ax2.set_title('Bias Y (Noise)')
ax2.legend()

plt.tight_layout()  # 调整子图间距
plt.show()