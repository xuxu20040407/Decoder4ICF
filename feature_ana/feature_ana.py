import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
import os

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

def calculate_distribution(data,index_vector):
    labels = data[:, 100:][:,index_vector]
    mean_values = np.mean(labels, axis=0)
    std_values = np.std(labels, axis=0)
    return mean_values, std_values

def train_and_analyze_model(data_path, original_data_path, index=None, retrain=False, num_epochs=300, lr=0.001):
    """
    训练解码器模型并进行分析。

    参数:
        data_path (str): 数据文件路径。
        original_data_path (str): 原始数据文件路径。
        index (int, optional): 模型索引和数据索引。默认为None。
        retrain (bool, optional): 是否重新训练模型。默认为False。
        num_epochs (int, optional): 训练轮数。默认为300。
        lr (float, optional): 学习率。默认为0.001。
    """
    # 定义模型结构
    input_size = 100
    hidden_size = 64
    feature_size = 5

    # 加载数据
    data = np.load(data_path)
    original_data = np.load(original_data_path)

    # 提取 index_data
    if index is not None:
        index_data = data[original_data.shape[0] + index * 3125:original_data.shape[0] + (index + 1) * 3125]
    else:
        index_data = data

    # 计算 index_data 的均值和标准差
    index_mean, index_std = calculate_distribution(index_data, [0, 1, 2, 3, 5])

    # 提取标签和功率数据
    index_label=['rhoR','Tmean','Vimplo','IFAR','alphaDT','rhomax',
    't','Eimplo','Edelas']
    index_vector = [0, 1, 2, 3, 5]
    powers = data[:, :100]
    labels = data[:, 100:][:, index_vector]

    original_powers = original_data[:, :100]
    original_labels = original_data[:, 100:][:, index_vector]

    # 使用 MinMaxScaler 进行归一化
    scaler_powers = MinMaxScaler()
    scaler_labels = MinMaxScaler()

    normalized_original_powers = scaler_powers.fit_transform(original_powers)
    normalized_original_labels = scaler_labels.fit_transform(original_labels)

    normalized_powers = scaler_powers.transform(powers)
    normalized_labels = scaler_labels.transform(labels)

    # 转换为 PyTorch 张量
    X_train_tensor = torch.tensor(normalized_labels, dtype=torch.float32)  # 输入：归一化后的标签
    y_train_tensor = torch.tensor(normalized_powers, dtype=torch.float32)  # 输出：归一化后的功率序列

    # 初始化解码器
    decoder = Decoder(feature_size, hidden_size, input_size)

    # 如果不重新训练且存在预训练模型，则加载模型
    if not retrain and index is not None:
        model_path = f"./decoder_model/decoder_model_{index}.pth"
        if os.path.exists(model_path):
            print(f"加载预训练模型: {model_path}")
            decoder.load_state_dict(torch.load(model_path))
        else:
            print(f"未找到模型文件: {model_path}，将重新训练模型。")
            retrain = True

    # 如果需要重新训练模型
    if retrain:
        print("开始训练模型...")
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(decoder.parameters(), lr=lr)

        # 训练模型
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

    # 定义偏移系数（基于标准差）
    offsets = [-1, -0.5,0, 0.5, 1]

    # 创建画布
    fig, axes = plt.subplots(feature_size, 5, figsize=(20, 20))  # feature_size行5列的子图

    # 遍历每个标签和偏移
    for label_idx in range(feature_size):  # feature_size个标签
        for offset_idx, offset in enumerate(offsets):  # 5个偏移
            # 修改当前标签的值（在原始数据上操作）
            modified_labels = first_sample_labels.copy()
            modified_labels[label_idx] += offset * index_std[label_idx]

            # 将修改后的标签归一化
            normalized_modified_labels = scaler_labels.transform(modified_labels.reshape(1, -1))

            # 使用解码器重构功率序列
            modified_labels_tensor = torch.tensor(normalized_modified_labels, dtype=torch.float32)
            with torch.no_grad():
                reconstructed_power = decoder(modified_labels_tensor).numpy().flatten()

            # 反归一化重构的功率序列
            reconstructed_power_inv = scaler_powers.inverse_transform(reconstructed_power.reshape(1, -1)).flatten()

            # 绘制原始功率曲线和重构功率曲线
            ax = axes[label_idx, offset_idx]
            ax.plot(first_sample_powers, label="Original Power", color="blue")
            ax.plot(reconstructed_power_inv, label=f"Reconstructed Power (Offset: {offset:.2f} std)", color="orange", linestyle="--")
            ax.set_title(f" {index_label[index_vector[label_idx]]},  {index_mean[label_idx]+offset*index_std[label_idx]:.4f}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Power")
            ax.legend()

    # 调整子图间距
    plt.tight_layout()
    plt.savefig("./feature_ana/label_modification_analysis.png")
    plt.show()
    print("分析完成，结果已保存到 label_modification_analysis.png")


# train_and_analyze_model(data_path="combined_data.npy",original_data_path="original_data.npy", retrain=True)
train_and_analyze_model(data_path="combined_data.npy",original_data_path="original_data1.npy", index=2)