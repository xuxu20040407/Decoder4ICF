import itertools
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pymulti as pm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pymulti as pm
import os

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_distribution(data,index_vector):
    labels = data[:, 100:][:,index_vector]
    powers = data[:, 0:100]
    scaler_label = MinMaxScaler()
    labels_normalized = scaler_label.fit_transform(labels)
    scaler_power = MinMaxScaler()
    power_normalized = scaler_power.fit_transform(powers)
    mean_values = np.mean(labels, axis=0)
    std_values = np.std(labels, axis=0)
    return mean_values, std_values, scaler_label,scaler_power

def generate_power_sequences(mean_values, std_values, decoder, scaler_power, scaler_label, index_vector, num_samples=5):
    samples = []

    for i in range(len(index_vector)):
        sample_values = np.linspace(mean_values[i] - 0.2 * std_values[i], mean_values[i] + 0.2 * std_values[i], num_samples)
        samples.append(sample_values)


    all_combinations = list(itertools.product(*samples))

    normalized_combinations = scaler_label.transform(all_combinations)

    sample_tensor = torch.tensor(normalized_combinations, dtype=torch.float32)

    decoder.eval()
    with torch.no_grad():
        generated_power_sequences = decoder(sample_tensor)

    generated_power_sequences = scaler_power.inverse_transform(generated_power_sequences.numpy())
    generated_power_sequences[generated_power_sequences < 0] = 0

    return generated_power_sequences

def train_decoder(decoder, index, data,scaler_label,scaler_power,index_vector, num_epochs=100, batch_size=32, lr=0.001):
    criterion = nn.MSELoss()
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=lr)

    power_sequences = scaler_power.fit_transform(data[:, :100])
    labels = scaler_label.fit_transform(data[:, 100:][:,index_vector])

    power_sequences_tensor = torch.tensor(power_sequences, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(power_sequences_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_path = f"decoder_model_{index-1}.pth"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, weights_only=True)
        decoder.load_state_dict(state_dict)
        print(f"Loaded existing model from {model_path}")

    for epoch in range(num_epochs):
        decoder.train()
        for batch in dataloader:
            power_sequences, true_labels = batch
            optimizer_decoder.zero_grad()
            reconstructed_power = decoder(true_labels)
            loss = criterion(reconstructed_power, power_sequences)
            loss.backward()
            optimizer_decoder.step()

        if epoch % 10 == 9:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    new_model_path = f"./decoder_model/decoder_model_{index}.pth"
    torch.save(decoder.state_dict(), new_model_path)
    return 

def power_reconstruction(decoder, index, index_vector, scaler_label, scaler_power):
    # data_path = "original_data1.npy"
    data_path = "LHS_data.npy"
    data = np.load(data_path)

    power_sequences = data[:, :100]
    labels = data[:, 100:][:, index_vector]

    # 使用已有的 scaler 进行归一化
    power_sequences = scaler_power.transform(power_sequences)
    labels = scaler_label.transform(labels)

    power_sequences_tensor = torch.tensor(power_sequences, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    with torch.no_grad():
        decoder.eval()  # 切换到评估模式
        first_sample_power = power_sequences_tensor[0:1]
        first_sample_label = labels_tensor[0:1]
        reconstructed_power = decoder(first_sample_label).numpy().flatten()

    # 反归一化
    original_power = scaler_power.inverse_transform(first_sample_power.numpy().reshape(1, -1)).flatten()
    reconstructed_power = scaler_power.inverse_transform(reconstructed_power.reshape(1, -1)).flatten()

    # 绘制原始功率序列和重构功率序列的对比图
    plt.figure(figsize=(12, 6))
    plt.plot(original_power, label="Original Power", color="blue")
    plt.plot(reconstructed_power, label="Reconstructed Power", color="orange", linestyle="--")
    plt.title(f"Original vs Reconstructed Power Sequence (Iteration {index})")
    plt.xlabel("Time Step")
    plt.ylabel("Power")
    plt.legend()
    plt.savefig(f"./power_reconstruction/power_reconstruction_{index}.png")
    plt.close()

    return 

def plot_all_laser(data, index):
    laser = data[:, :100]
    time = np.arange(100) * 5.69 / 100  # Convert range to numpy array before multiplication
    plt.plot(time, laser.T)  # Transpose laser to match time dimensions
    plt.title(f"Laser")
    plt.xlabel("time")
    plt.ylabel("power")
    plt.tight_layout()
    plt.savefig(f"./data_laser/data_laser_{index}.png")
    plt.close()

def plot_distribution(data, index,index_vector):
    feature = data[:, 100:][:,index_vector]
    means = np.mean(feature, axis=0)
    stds = np.std(feature, axis=0)
    output_size=len(index_vector)

    plt.figure(figsize=(20, 20/output_size))
    for i in range(feature.shape[1]):
        plt.subplot(1, output_size, i + 1)
        plt.hist(feature[:, i], bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f"Label {i}: mean={means[i]:.2f}, std={stds[i]:.2f}")
        plt.xlabel("data")
        plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(f"./data_distribution/data_distribution_{index}.png")
    plt.close()

def process_task(index, new_dir,Laser_grid):
    pm.generate_input_data1D(new_dir, index,Laser_grid[index])
    pm.run_command_1D(new_dir, index)

def thread_task(index, new_dir, Laser_grid):
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(process_task, index, new_dir, Laser_grid)

def save_npy(old_data,num_samples,output_size):
    program_name='Multi-1D'
    inp_data=[]
    fit_data=[]
    for i in range(num_samples**output_size):
        inp_data.append(pm.data1D_process_inp(program_name,i))
        fit_data.append(pm.data1D_process_fit(program_name, i))
    inp_data=np.array(inp_data)
    fit_data=np.array(fit_data)
    new_data = np.concatenate((
        inp_data[:, 100:200],  # 选择输入数据中的特定列
        fit_data[:, :]   # 选择拟合数据中的特定列
    ), axis=1)
    final_data= np.concatenate((old_data,new_data), axis=0)
    print("save end")
    return final_data,new_data


input_size = 100
hidden_size = 64
index_vector = [0, 1,2,3,5]
output_size = len(index_vector)
num_samples=3


decoder = Decoder(output_size, hidden_size, input_size)

# data = np.load("original_data1.npy")
data = np.load("LHS_data.npy")
plot_distribution(data, -1,index_vector)
plot_all_laser(data, -1)
program_name = 'Multi-1D'
new_dir = pm.init1D(program_name)

mean_values, std_values, scaler_label,scaler_power = calculate_distribution(data,index_vector)

for iteration in range(3):
    print(f"Iteration {iteration + 1} started.")
    
    # 训练模型
    train_decoder(decoder, iteration, data,scaler_label,scaler_power,index_vector, num_epochs=100)
    power_reconstruction(decoder,iteration,index_vector,scaler_label,scaler_power)
    
    # 生成数据
    Laser_grid = generate_power_sequences(mean_values, std_values, decoder, scaler_power, scaler_label, index_vector,num_samples=num_samples)
    print(f"Generated {Laser_grid.shape[0]} power sequences.")
    
    # 创建扩展配置
    new_column = np.full((Laser_grid.shape[0], 100), 0.05747)
    new_column[:, 0] = 0
    Laser_grid = np.hstack((new_column, Laser_grid))
    
    # 多进程处理
    with ProcessPoolExecutor(max_workers=60) as pool:
        futures = [pool.submit(thread_task, i, new_dir, Laser_grid) for i in range(Laser_grid.shape[0])]
    
    # 保存扩展数据
    data,new_data=save_npy(data,num_samples,output_size)
    # 绘制分布图
    plot_distribution(new_data, iteration,index_vector)
    plot_all_laser(new_data, iteration)

plot_distribution(data, 999,index_vector)
plot_all_laser(data, 999)

np.save("combined_data", data)