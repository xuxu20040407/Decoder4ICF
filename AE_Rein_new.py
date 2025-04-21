from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pymulti as pm
from concurrent.futures import ProcessPoolExecutor
import itertools

# ======================
# 配置常量
# ======================
class Config:
    # 路径配置
    DIRS = {
        "models": Path("./ae_models"),
        "reconstructions": Path("./ae_reconstructions"),
        "latent": Path("./ae_data_latent_distribution"),
        "label": Path("./ae_data_distribution"),
        "lasers": Path("./ae_data_laser")
    }
    
    # 模型参数
    INPUT_SIZE = 100
    LATENT_SIZE = 5
    HIDDEN_SIZE = 64
    
    # 训练参数
    NUM_EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    # 数据参数
    NUM_SAMPLES = 5
    NUM_ITERATIONS = 3
    PROGRAM_NAME = "Multi-1D-AE"
    
    # 文件路径
    INIT_DATA_PATH = "LHS_data.npy"
    OUTPUT_DATA_PATH = "ae_combined_data.npy"

# ======================
# 初始化目录
# ======================
for d in Config.DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# ======================
# 模型定义
# ======================
class AutoEncoder(nn.Module):
    """自编码器网络结构"""
    def __init__(self, input_size: int = Config.INPUT_SIZE, 
                 latent_size: int = Config.LATENT_SIZE,
                 hidden_size: int = Config.HIDDEN_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

# ======================
# 核心功能模块
# ======================
def prepare_data(data: np.ndarray, scaler: MinMaxScaler = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """数据预处理与归一化"""
    scaler = scaler or MinMaxScaler()
    if not scaler.n_samples_seen_:
        normalized = scaler.fit_transform(data)
    else:
        normalized = scaler.transform(data)
    return normalized, scaler

def train_autoencoder(
    model: AutoEncoder,
    data: np.ndarray,
    iteration: int,
    scaler: MinMaxScaler
) -> Tuple[AutoEncoder, float]:
    """训练自编码器并保存模型"""
    # 数据准备
    normalized, _ = prepare_data(data, scaler)
    dataset = TensorDataset(torch.tensor(normalized, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # 训练配置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 训练循环
    model.train()
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 9:
            avg_loss = total_loss / len(dataloader)
            print(f"Iter [{iteration}] Epoch [{epoch+1}/{Config.NUM_EPOCHS}] Loss: {avg_loss:.4f}")
    
    # 保存模型
    model_path = Config.DIRS["models"] / f"autoencoder_{iteration}.pth"
    torch.save(model.state_dict(), model_path)
    
    return model, avg_loss

def analyze_latent_space(
    model: AutoEncoder,
    data: np.ndarray,
    scaler: MinMaxScaler
) -> Tuple[np.ndarray, np.ndarray]:
    """计算潜在空间统计特征"""
    normalized = scaler.transform(data)
    with torch.no_grad():
        latent = model.encoder(torch.tensor(normalized, dtype=torch.float32))
    return latent.mean(dim=0).numpy(), latent.std(dim=0).numpy()

def generate_latent_variations(
    mean: np.ndarray, 
    std: np.ndarray, 
    num_samples: int = Config.NUM_SAMPLES
) -> np.ndarray:
    """生成潜在空间采样点"""
    variations = [
        np.linspace(m - 0.5*s, m + 0.5*s, num_samples)
        for m, s in zip(mean, std)
    ]
    return np.array(list(itertools.product(*variations)))

def generate_power_sequences(
    model: AutoEncoder,
    latent_samples: np.ndarray,
    scaler: MinMaxScaler
) -> np.ndarray:
    """从潜在空间生成功率序列"""
    with torch.no_grad():
        generated = model.decoder(torch.tensor(latent_samples, dtype=torch.float32))
    sequences = scaler.inverse_transform(generated.numpy())
    return np.clip(sequences, 0, None)

# ======================
# 可视化模块
# ======================
def visualize_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    iteration: int
) -> None:
    """可视化重建效果对比"""
    plt.figure(figsize=(12, 6))
    plt.plot(original, label="Original", alpha=0.8)
    plt.plot(reconstructed, label="Reconstructed", linestyle="--", alpha=0.8)
    plt.title(f"Reconstruction Comparison (Iteration {iteration})")
    plt.xlabel("Time Step")
    plt.ylabel("Power")
    plt.legend()
    
    save_path = Config.DIRS["reconstructions"] / f"recon_{iteration}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def visualize_latent_distribution(
    latent_data: np.ndarray,
    iteration: int
) -> None:
    """可视化潜在空间分布"""
    plt.figure(figsize=(20, 5))
    for i in range(latent_data.shape[1]):
        ax = plt.subplot(1, Config.LATENT_SIZE, i+1)
        ax.hist(latent_data[:, i], bins=30, color='green', alpha=0.7, edgecolor='black')
        ax.set_title(f"Latent Dim {i}\nμ={latent_data[:,i].mean():.2f} σ={latent_data[:,i].std():.2f}")
        ax.tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    save_path = Config.DIRS["latent"] / f"latent_dist_{iteration}.png"
    plt.savefig(save_path)
    plt.close()



def plot_all_laser(data: np.ndarray, iteration: int) -> None:
    """绘制所有激光时序图 (原plot_all_laser)"""
    laser = data[:, :Config.INPUT_SIZE]
    time = np.linspace(0, 5.69, Config.INPUT_SIZE)  # 更精确的时间生成方式
    
    plt.figure(figsize=(12, 6))
    
    # 优化绘图参数
    alpha = 0.3 if laser.shape[0] > 50 else 0.7  # 样本过多时降低透明度
    linewidth = 0.5 if laser.shape[0] > 100 else 1.0
    
    for i in range(laser.shape[0]):
        plt.plot(time, laser[i], 
                color='blue', 
                alpha=alpha,
                linewidth=linewidth)
    
    # 添加统计参考线
    mean_line = np.mean(laser, axis=0)
    plt.plot(time, mean_line, 
            color='red', 
            linewidth=2, 
            linestyle='--',
            label='Mean')
    
    plt.title(f"Laser Power Sequences (N={laser.shape[0]})")
    plt.xlabel("Time (ns)")
    plt.ylabel("Power (MW)")
    plt.legend()
    
    # 优化坐标轴
    plt.xlim(0, 5.7)
    plt.ylim(bottom=0)  # 功率不为负
    
    save_path = Config.DIRS["lasers"] / f"data_laser_{iteration}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def visualize_label_distribution(
    data: np.ndarray,
    iteration: int,
    index_vector=[0,1,2,3,5]
) -> None:
    print(data.shape)
    label = data[:, Config.INPUT_SIZE:][:,index_vector]
    plt.figure(figsize=(20, 5))
    for i in range(label.shape[1]):
        ax = plt.subplot(1, 5, i+1)
        ax.hist(label[:, i], bins=30, color='green', alpha=0.7, edgecolor='black')
        ax.set_title(f"Latent Dim {i}\nμ={label[:,i].mean():.2f} σ={label[:,i].std():.2f}")
        ax.tick_params(axis='both', labelsize=8)
    
    plt.tight_layout()
    save_path = Config.DIRS["label"] / f"latent_dist_{iteration}.png"
    plt.savefig(save_path)
    plt.close()

# ======================
# 数据处理模块
# ======================
def prepare_simulation_input(generated_power: np.ndarray) -> np.ndarray:
    """准备仿真输入格式"""
    base = np.full((generated_power.shape[0], Config.INPUT_SIZE), 0.05747)
    base[:, 0] = 0
    return np.hstack((base, generated_power))

def process_simulation_task(index: int, base_dir: Path, laser_data: np.ndarray) -> None:
    """执行单个仿真任务"""
    pm.generate_input_data1D(str(base_dir), index, laser_data[index])
    pm.run_command_1D(str(base_dir), index)

def collect_simulation_results(program_name: str, num_samples: int) -> np.ndarray:
    """收集仿真结果数据"""
    results = []
    for i in range(num_samples**Config.LATENT_SIZE):
        inp = pm.data1D_process_inp(program_name, i)[Config.INPUT_SIZE:Config.INPUT_SIZE*2]
        fit = pm.data1D_process_fit(program_name, i)
        results.append(np.concatenate([inp,fit],axis=0))
    return np.array(results)

# ======================
# 主工作流程
# ======================
def main_workflow():
    # 初始化
    initial_data = np.load(Config.INIT_DATA_PATH)
    power_data = initial_data[:, :Config.INPUT_SIZE]
    all_data=initial_data
    scaler = MinMaxScaler().fit(power_data)
    
    model = AutoEncoder()
    simulation_dir = pm.init1D(Config.PROGRAM_NAME)

    plot_all_laser(initial_data, iteration=-1)
    visualize_label_distribution(initial_data, iteration=-1)
    
    # 迭代流程
    for iteration in range(Config.NUM_ITERATIONS):
        print(f"\n=== Iteration {iteration + 1}/{Config.NUM_ITERATIONS} ===")
        
        # 1. 训练自编码器
        model, loss = train_autoencoder(model, power_data, iteration, scaler)
        
        # 2. 重建可视化
        sample = power_data[:1]
        normalized = scaler.transform(sample)
        with torch.no_grad():
            reconstructed = model(torch.tensor(normalized, dtype=torch.float32)).numpy()
        visualize_reconstruction(scaler.inverse_transform(normalized).flatten(), 
                              scaler.inverse_transform(reconstructed).flatten(), 
                              iteration)
        
        # 3. 生成新样本
        print(f"\n=== Iteration {iteration+1}/{Config.NUM_ITERATIONS} Generated===")
        mean, std = analyze_latent_space(model, power_data, scaler)
        latent_samples = generate_latent_variations(mean, std)
        new_power = generate_power_sequences(model, latent_samples, scaler)
        
        # 4. 准备仿真输入
        simulation_input = prepare_simulation_input(new_power)
        
        # 5. 并行执行仿真
        with ProcessPoolExecutor(max_workers=50) as executor:
            tasks = [executor.submit(process_simulation_task, i, simulation_dir, simulation_input)
                    for i in range(len(simulation_input))]
        
        # 6. 收集并合并数据
        new_data = collect_simulation_results(Config.PROGRAM_NAME, Config.NUM_SAMPLES)
        power_data = np.concatenate((power_data, new_data[:,:Config.INPUT_SIZE]), axis=0)
        all_data = np.concatenate((all_data, new_data), axis=0)
        
        # 7. 潜在空间可视化
        with torch.no_grad():
            latent = model.encoder(torch.tensor(scaler.transform(new_data[:,:Config.INPUT_SIZE]), dtype=torch.float32))
        visualize_latent_distribution(latent.numpy(), iteration)
        plot_all_laser(new_data, iteration=iteration)
        visualize_label_distribution(new_data, iteration=iteration)

    
    # 最终保存
    np.save(Config.OUTPUT_DATA_PATH, all_data)
    print("Workflow completed successfully!")

if __name__ == "__main__":
    main_workflow()