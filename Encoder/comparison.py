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

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size=100, d_model=64, nhead=4, num_layers=2, output_size=5):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model * input_size, 64),
            nn.Tanh(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)
        transformer_out = transformer_out.permute(1, 0, 2)
        batch_size = transformer_out.shape[0]
        x = transformer_out.reshape(batch_size, -1)
        x = self.fc(x)
        return x

# 参数初始化
input_size = 100
hidden_size = 64
output_size = 5

# 加载数据
data = np.load("combined_data.npy")
ae_data = np.load("ae_combined_data.npy")
original_data = np.load("original_data1.npy")
LHS_data = np.load("LHS_data.npy")
original_size = original_data.shape[0]
index_vector = [0, 1, 2, 3, 5]

# 数据预处理
test_data = np.concatenate([data[original_size:], ae_data[original_size:]], axis=0)
random_indices = np.random.choice(len(test_data), size=2000, replace=False)
random_test_data = test_data[random_indices]

def extract_sequences_and_labels(data, index_vector):
    power_sequences = data[:, :100]
    labels = data[:, 100:][:, index_vector]
    return power_sequences, labels

def data_process(train_data, test_data):
    scaler = MinMaxScaler()
    scaler_test = MinMaxScaler()
    train_normalized = scaler.fit_transform(train_data)
    test_normalized = scaler.transform(test_data)
    scaler_test.fit(train_data[:, 100:][:, index_vector])
    train_power, train_labels = extract_sequences_and_labels(train_normalized, index_vector)
    test_power, test_labels = extract_sequences_and_labels(test_normalized, index_vector)
    return (torch.tensor(train_power, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.float32),
            torch.tensor(test_power, dtype=torch.float32),
            torch.tensor(test_labels, dtype=torch.float32),
            scaler_test)

# 处理各数据集
train_data_1 = original_data
train_data_2 = data[-original_size:]
train_data_3 = ae_data[-original_size:]
train_data_4 = LHS_data[-original_size:]

datasets = [
    data_process(train_data_1, random_test_data),
    data_process(train_data_2, random_test_data),
    data_process(train_data_3, random_test_data),
    data_process(train_data_4, random_test_data)
]

# 训练函数
def train_model(model, train_power, train_labels, num_epochs=1):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(train_power, train_labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            x, y = batch
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.MSELoss()(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return losses

# 评估函数
def evaluate(model, test_power, test_labels, scaler):
    model.eval()
    with torch.no_grad():
        pred = model(test_power)
        pred = scaler.inverse_transform(pred.numpy())
        true = scaler.inverse_transform(test_labels.numpy())
        return true, pred

# 训练并评估各模型
results = []

for i, (train_power, train_labels, test_power, test_labels, scaler) in enumerate(datasets):
    # 训练Encoder
    encoder = Encoder(input_size, hidden_size, output_size)
    enc_loss = train_model(encoder, train_power, train_labels)
    enc_true, enc_pred = evaluate(encoder, test_power, test_labels, scaler)
    
    # 训练Transformer
    transformer = TransformerModel()
    trans_loss = train_model(transformer, train_power, train_labels)
    trans_true, trans_pred = evaluate(transformer, test_power, test_labels, scaler)
    
    results.append({
        "name": ["Original", "Physics-Informed", "AutoEncoder", "LHS"][i],
        "encoder": (enc_true, enc_pred, enc_loss),
        "transformer": (trans_true, trans_pred, trans_loss)
    })

# 定义颜色和数据集名称
dataset_colors = ['blue', 'green', 'red', 'purple']
dataset_names = ["Original", "Physics-Informed", "AutoEncoder", "LHS"]

# 绘制第一张图：2x5子图（Encoder vs Transformer）
plt.figure(figsize=(25, 10))
for label in range(5):
    # Encoder行
    plt.subplot(2, 5, label+1)
    enc_true, enc_pred = results[0]["encoder"][0], results[0]["encoder"][1]
    plt.scatter(enc_true[:, label], enc_pred[:, label], alpha=0.3, c=dataset_colors[0])
    min_val = min(enc_true[:, label].min(), enc_pred[:, label].min())
    max_val = max(enc_true[:, label].max(), enc_pred[:, label].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f"Encoder - Label {label+1}")
    
    # Transformer行
    plt.subplot(2, 5, label+6)
    trans_true, trans_pred = results[0]["transformer"][0], results[0]["transformer"][1]
    plt.scatter(trans_true[:, label], trans_pred[:, label], alpha=0.3, c=dataset_colors[0])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(f"Transformer - Label {label+1}")

plt.tight_layout()
plt.savefig("./Encoder/model_comparison.png", dpi=300)
plt.close()

# 绘制第二张图：4x5子图（数据集对比）
plt.figure(figsize=(25, 20))
for ds in range(4):
    for label in range(5):
        plt.subplot(4, 5, ds*5 + label +1)
        # 绘制Encoder结果
        enc_true = results[ds]["encoder"][0][:, label]
        enc_pred = results[ds]["encoder"][1][:, label]
        plt.scatter(enc_true, enc_pred, alpha=0.3, c=dataset_colors[ds], label=f'{dataset_names[ds]} - Encoder')
        
        # 绘制Transformer结果
        trans_true = results[ds]["transformer"][0][:, label]
        trans_pred = results[ds]["transformer"][1][:, label]
        plt.scatter(trans_true, trans_pred, alpha=0.3, c=dataset_colors[ds], marker='x', label=f'{dataset_names[ds]} - Transformer')
        
        # 绘制参考线
        min_val = min(enc_true.min(), trans_true.min())
        max_val = max(enc_true.max(), trans_true.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f"{dataset_names[ds]} - Label {label+1}")
        plt.legend()

plt.tight_layout()
plt.savefig("./Encoder/dataset_comparison.png", dpi=300)
plt.close()