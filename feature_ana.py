import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义模型结构
input_size = 100
hidden_size = 64
feature_size = 5

# 加载数据
data_path = "combined_data.npy"
data = np.load(data_path)
index_vector = [0, 1, 2, 3, 5]
powers = data[:, :100]
labels = data[:, 100:][:,index_vector]

# 使用 MinMaxScaler 进行归一化
scaler_powers = MinMaxScaler()
scaler_labels = MinMaxScaler()

normalized_powers = scaler_powers.fit_transform(powers)
normalized_labels = scaler_labels.fit_transform(labels)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(normalized_labels, dtype=torch.float32)  # 输入：归一化后的标签
y_train_tensor = torch.tensor(normalized_powers, dtype=torch.float32)  # 输出：归一化后的功率序列

# 初始化解码器
decoder = Decoder(feature_size, hidden_size, input_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(decoder.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = decoder(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

decoder.eval()

# 提取第一行数据
first_sample_labels = labels[0]  # 第一行原始标签
first_sample_powers = powers[0]  # 第一行原始功率序列

# 定义系数范围
coefficients = [0.8, 0.9, 1.0, 1.1, 1.2]

# 创建画布
fig, axes = plt.subplots(feature_size, 5, figsize=(20, 20))  # feature_size行5列的子图

# 遍历每个标签和系数
for label_idx in range(feature_size):  # feature_size个标签
    for coeff_idx, coeff in enumerate(coefficients):  # 5个系数
        # 修改当前标签的值（在原始数据上操作）
        modified_labels = first_sample_labels.copy()
        modified_labels[label_idx] *= coeff

        # 将修改后的标签归一化
        normalized_modified_labels = scaler_labels.transform(modified_labels.reshape(1, -1))

        # 使用解码器重构功率序列
        modified_labels_tensor = torch.tensor(normalized_modified_labels, dtype=torch.float32)
        with torch.no_grad():
            reconstructed_power = decoder(modified_labels_tensor).numpy().flatten()

        # 反归一化重构的功率序列
        reconstructed_power_inv = scaler_powers.inverse_transform(reconstructed_power.reshape(1, -1)).flatten()

        # 绘制原始功率曲线和重构功率曲线
        ax = axes[label_idx, coeff_idx]
        ax.plot(first_sample_powers, label="Original Power", color="blue")
        ax.plot(reconstructed_power_inv, label=f"Reconstructed Power (Coeff: {coeff})", color="orange", linestyle="--")
        ax.set_title(f"Label {label_idx + 1}, Coeff {coeff}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Power")
        ax.legend()

# 调整子图间距
plt.tight_layout()
plt.savefig("label_modification_analysis.png")
