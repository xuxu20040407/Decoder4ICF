import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 定义编码器模型
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# 初始化模型参数
input_size = 100  # 功率序列的长度
hidden_size = 64  # 隐藏层大小
output_size = 5   # 标签的长度

# 加载数据
data_path = "combined_data.npy"  # 替换为您的 combined_data.npy 文件路径
data = np.load(data_path)  # 加载 combined_data.npy 文件

ae_data_path = "ae_combined_data.npy"  # 替换为您的 ae_combined_data.npy 文件路径
ae_data = np.load(ae_data_path)  # 加载 ae_combined_data.npy 文件

original_data_path = "original_data1.npy"  # 替换为您的 original_data1.npy 文件路径
original_data = np.load(original_data_path)
original_size = original_data.shape[0]

LHS_data_path = "LHS_data.npy" 
LHS_data = np.load(LHS_data_path)

index_vector = [0, 1, 2, 3, 5]

# 分割数据集
train_data_1 = original_data
train_data_2 = data[-original_size:]  # 第二组训练数据
train_data_3 = ae_data[-original_size:]  # 使用与 original_data 相同大小的训练数据
train_data_4 = LHS_data[-original_size:]  # 使用与 original_data 相同大小的训练数据

test_data=np.concatenate([data[original_size:],ae_data[original_size:]],axis=0)
# test_data = data[original_size:-original_size]  # 测试数据
random_indices = np.random.choice(len(test_data), size=2000, replace=False)
random_test_data = test_data[random_indices]

# 提取功率序列和标签
def extract_sequences_and_labels(data, index_vector):
    power_sequences = data[:, :100]
    labels = data[:, 100:][:, index_vector]
    return power_sequences, labels


def data_process(train_data_1,random_test_data):
    scaler_1 = MinMaxScaler()
    scaler_test_1 = MinMaxScaler()
    train_data_1_normalized = scaler_1.fit_transform(train_data_1)
    test_data_normalized_1 = scaler_1.transform(random_test_data)
    scaler_test_1.fit_transform(train_data_1[:, 100:][:, index_vector])
    train_power_sequences_1, train_labels_1 = extract_sequences_and_labels(train_data_1_normalized, index_vector)
    test_power_sequences_1, test_labels_1 = extract_sequences_and_labels(test_data_normalized_1, index_vector)
    train_power_sequences_tensor_1 = torch.tensor(train_power_sequences_1, dtype=torch.float32)
    train_labels_tensor_1 = torch.tensor(train_labels_1, dtype=torch.float32)
    test_power_sequences_tensor_1 = torch.tensor(test_power_sequences_1, dtype=torch.float32)
    test_labels_tensor_1 = torch.tensor(test_labels_1, dtype=torch.float32)
    return train_power_sequences_tensor_1,train_labels_tensor_1,test_power_sequences_tensor_1,test_labels_tensor_1,scaler_test_1


train_power_sequences_tensor_1,train_labels_tensor_1,test_power_sequences_tensor_1,test_labels_tensor_1,scaler_test_1=data_process(train_data_1,random_test_data)
train_power_sequences_tensor_2,train_labels_tensor_2,test_power_sequences_tensor_2,test_labels_tensor_2,scaler_test_2=data_process(train_data_2,random_test_data)
train_power_sequences_tensor_3,train_labels_tensor_3,test_power_sequences_tensor_3,test_labels_tensor_3,scaler_test_3=data_process(train_data_3,random_test_data)
train_power_sequences_tensor_4,train_labels_tensor_4,test_power_sequences_tensor_4,test_labels_tensor_4,scaler_test_4=data_process(train_data_4,random_test_data)


# 定义损失函数
criterion = nn.MSELoss()

# 训练模型的函数
def train_model(train_power_sequences, train_labels, num_epochs=500):
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

# 训练三个模型
print("Training Model 1 with original_data...")
model_1, losses_1 = train_model(train_power_sequences_tensor_1, train_labels_tensor_1)

print("Training Model 2 with loop_data...")
model_2, losses_2 = train_model(train_power_sequences_tensor_2, train_labels_tensor_2)

print("Training Model 3 with ae_combined_data...")
model_3, losses_3 = train_model(train_power_sequences_tensor_3, train_labels_tensor_3)

print("Training Model 4 with ae_combined_data...")
model_4, losses_4 = train_model(train_power_sequences_tensor_4, train_labels_tensor_4)

# 绘制对数损失下降图
plt.figure(figsize=(10, 5))
plt.semilogy(losses_1, label="Original Loss")  # 使用 semilogy 绘制对数图
plt.semilogy(losses_2, label="Physics-Informed Loss")  # 使用 semilogy 绘制对数图
plt.semilogy(losses_3, label="AutoEncoder Loss")  # 添加 Model 3 的损失曲线
plt.semilogy(losses_4, label="LHS Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Loss During Training for Different Data Segments (Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # 添加网格线
plt.savefig("./Encoder/loss_log.png",dpi=300)
plt.close()

# 在测试集上评估模型
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

print("Evaluating Model 3 on test data...")
true_labels_3, predicted_labels_3 = evaluate_model(model_3, test_power_sequences_tensor_3, test_labels_tensor_3, scaler_test_3)

print("Evaluating Model 4 on test data...")
true_labels_4, predicted_labels_4 = evaluate_model(model_4, test_power_sequences_tensor_4, test_labels_tensor_4, scaler_test_4)

# 绘制真实标签与预测标签的对比图
plt.figure(figsize=(20, 10))
for i in range(output_size):
    plt.subplot(1, output_size, i + 1)
    plt.scatter(true_labels_1[:, i], predicted_labels_1[:, i], alpha=0.3, label="Original Dataset")
    plt.scatter(true_labels_2[:, i], predicted_labels_2[:, i], alpha=0.3, label="Physics-Informed Dataset")
    plt.scatter(true_labels_3[:, i], predicted_labels_3[:, i], alpha=0.3, label="AutoEncoder Dataset")
    plt.scatter(true_labels_4[:, i], predicted_labels_4[:, i], alpha=0.3, label="LHS Dataset")
    plt.plot([min(true_labels_1[:, i].min(), true_labels_2[:, i].min(), true_labels_3[:, i].min()), 
              max(true_labels_1[:, i].max(), true_labels_2[:, i].max(), true_labels_3[:, i].max())],
             [min(true_labels_1[:, i].min(), true_labels_2[:, i].min(), true_labels_3[:, i].min()), 
              max(true_labels_1[:, i].max(), true_labels_2[:, i].max(), true_labels_3[:, i].max())],
             color="red", linestyle="--", label="y=x")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.title(f"Label {i + 1}")
    plt.legend()
plt.tight_layout()
plt.savefig("./Encoder/label.png",dpi=300)
plt.close()


plt.figure(figsize=(25, 20))  # 增大画布尺寸以适应 4x5 子图

# 数据集和预测结果的列表
true_labels_list = [true_labels_1, true_labels_2, true_labels_3, true_labels_4]
predicted_labels_list = [predicted_labels_1, predicted_labels_2, predicted_labels_3, predicted_labels_4]
dataset_names = ["Original Dataset", "Physics-Informed Dataset", "AutoEncoder Dataset", "LHS Dataset"]

# 遍历每个数据集（行）
for i in range(4):  # 4 个数据集
    # 遍历每个输出（列）
    for j in range(output_size):  # 5 个输出
        plt.subplot(4, 5, i * 5 + j + 1)  # 计算子图位置
        plt.scatter(true_labels_list[i][:, j], predicted_labels_list[i][:, j], alpha=0.3, label=dataset_names[i])
        
        # 计算 y=x 线的范围
        min_val = min(true_labels_list[i][:, j].min(), predicted_labels_list[i][:, j].min())
        max_val = max(true_labels_list[i][:, j].max(), predicted_labels_list[i][:, j].max())
        
        # 绘制 y=x 线
        plt.plot([min_val, max_val], [min_val, max_val], 
                 color="red", linestyle="--", label="y=x")
        
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.title(f"{dataset_names[i]} - Output {j + 1}")
        plt.legend()

plt.tight_layout()  # 调整子图间距，避免重叠
plt.savefig("./Encoder/label_comparison_4x5.png",dpi=300)
plt.close()