from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# 定义更深层的编码器
# class Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
#         super(Encoder, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, hidden_size)
#         self.fc4 = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(dropout_rate)  # 添加 Dropout 层

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)  # 应用 Dropout
#         x = torch.relu(self.fc2(x))
#         x = self.dropout(x)  # 应用 Dropout
#         x = torch.relu(self.fc3(x))
#         x = self.dropout(x)  # 应用 Dropout
#         x = self.fc4(x)
#         return x

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

# 初始化模型参数
input_size = 100  # 功率序列的长度
hidden_size = 64  # 隐藏层大小
output_size = 5   # 标签的长度
dropout_rate = 0.2  # Dropout 比率

# 加载数据
data_path = "combined_data.npy"  # 替换为您的.npy文件路径
data = np.load(data_path)  # 加载.npy文件
index_vector = [0, 1, 2, 3, 5]


original_data_path="original_data1.npy"
original_data=np.load(original_data_path)
original_size=original_data.shape[0]

# 分割数据集
train_data_1 = original_data  # 第一组训练数据
train_data_2 = data[-original_size:]  # 第二组训练数据
test_data = data[original_size:-original_size]  # 测试数据
random_indices = np.random.choice(len(test_data), size=500, replace=False)
random_test_data = test_data[random_indices]

# 使用 MinMaxScaler 进行归一化
scaler_1 = MinMaxScaler()
scaler_2 = MinMaxScaler()
scaler_test_1=MinMaxScaler()
scaler_test_2=MinMaxScaler()


# 分别对训练数据和测试数据进行归一化
train_data_1_normalized = scaler_1.fit_transform(train_data_1)
train_data_2_normalized = scaler_2.fit_transform(train_data_2)
test_data_normalized_1=scaler_1.transform(random_test_data)
test_data_normalized_2=scaler_2.transform(random_test_data)

scaler_test_1.fit_transform(train_data_1[:,100:][:,index_vector])
scaler_test_2.fit_transform(train_data_2[:,100:][:,index_vector])

# 提取功率序列和标签
def extract_sequences_and_labels(data,index_vector):
    power_sequences = data[:, :100]
    labels = data[:, 100:][:, index_vector]
    return power_sequences, labels

# 准备训练数据
train_power_sequences_1, train_labels_1 = extract_sequences_and_labels(train_data_1_normalized,index_vector)
train_power_sequences_2, train_labels_2 = extract_sequences_and_labels(train_data_2_normalized,index_vector)

# 准备测试数据
test_power_sequences_1, test_labels_1 = extract_sequences_and_labels(test_data_normalized_1,index_vector)
test_power_sequences_2, test_labels_2 = extract_sequences_and_labels(test_data_normalized_2,index_vector)

# 将数据转换为 PyTorch 张量
train_power_sequences_tensor_1 = torch.tensor(train_power_sequences_1, dtype=torch.float32)
train_labels_tensor_1 = torch.tensor(train_labels_1, dtype=torch.float32)
train_power_sequences_tensor_2 = torch.tensor(train_power_sequences_2, dtype=torch.float32)
train_labels_tensor_2 = torch.tensor(train_labels_2, dtype=torch.float32)
test_power_sequences_tensor_1 = torch.tensor(test_power_sequences_1, dtype=torch.float32)
test_labels_tensor_1 = torch.tensor(test_labels_1, dtype=torch.float32)
test_power_sequences_tensor_2 = torch.tensor(test_power_sequences_2, dtype=torch.float32)
test_labels_tensor_2 = torch.tensor(test_labels_2, dtype=torch.float32)
# 定义损失函数
criterion = nn.MSELoss()

# 训练模型的函数
def train_model(train_power_sequences, train_labels, num_epochs=100):
    encoder = Encoder(input_size, hidden_size, output_size)
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    dataset = TensorDataset(train_power_sequences, train_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    losses = []  # 记录损失
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            power_sequences, true_labels = batch
            optimizer.zero_grad()
            predicted_labels = encoder(power_sequences)
            loss = criterion(predicted_labels, true_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return encoder, losses

# 训练两个模型
print("Training Model 1 with original_data...")
model_1, losses_1 = train_model(train_power_sequences_tensor_1, train_labels_tensor_1)

print("Training Model 2 with loop_data...")
model_2, losses_2 = train_model(train_power_sequences_tensor_2, train_labels_tensor_2)

# 绘制两次训练的损失下降图
plt.figure(figsize=(10, 5))
plt.plot(losses_1, label="Model 1 Loss")
plt.plot(losses_2, label="Model 2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss During Training for Different Data Segments")
plt.legend()
plt.savefig("./Encoder/loss.png")

# 在测试集上评估两个模型
def evaluate_model(model, test_power_sequences, test_labels, scaler):
    model.eval()
    with torch.no_grad():
        predicted_labels = model(test_power_sequences)
        loss = criterion(predicted_labels, test_labels)
        print(f"Test Loss: {loss.item():.4f}")
        # 反归一化
        predicted_labels = scaler.inverse_transform(predicted_labels.numpy())
        true_labels = scaler.inverse_transform(test_labels.numpy())
        return true_labels, predicted_labels

print("Evaluating Model 1 on test data...")
true_labels_1, predicted_labels_1 = evaluate_model(model_1, test_power_sequences_tensor_1, test_labels_tensor_1, scaler_test_1)

print("Evaluating Model 2 on test data...")
true_labels_2, predicted_labels_2 = evaluate_model(model_2, test_power_sequences_tensor_2, test_labels_tensor_2, scaler_test_2)

# 绘制真实标签与预测标签的对比图
plt.figure(figsize=(20, 10))
for i in range(output_size):
    plt.subplot(1, output_size, i + 1)
    plt.scatter(true_labels_1[:, i], predicted_labels_1[:, i], alpha=0.5, label="Model 1")
    plt.scatter(true_labels_2[:, i], predicted_labels_2[:, i], alpha=0.5, label="Model 2")
    plt.plot([min(true_labels_1[:, i].min(), true_labels_2[:, i].min()), max(true_labels_1[:, i].max(), true_labels_2[:, i].max())],
             [min(true_labels_1[:, i].min(), true_labels_2[:, i].min()), max(true_labels_1[:, i].max(), true_labels_2[:, i].max())],
             color="red", linestyle="--", label="y=x")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title(f"Label {i + 1}")
    plt.legend()

plt.tight_layout()
plt.savefig("./Encoder/label.png")