from concurrent.futures import ProcessPoolExecutor
import itertools
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pymulti as pm

# 常量定义
class Config:
    NUM_ITERATIONS=3
    program_name = 'Multi-1D'
    NEW_DIR = pm.init1D(program_name)
    INPUT_SIZE = 100
    HIDDEN_SIZE = 64
    OUTPUT_INDICES = [0, 1, 2, 3, 5]
    NUM_SAMPLES = 5
    DATA_PATHS = {
        "input": "LHS_data.npy",
        "output": "pid_combined_data.npy"
    }
    SAVE_DIRS = {
        "models": Path("./decoder_models"),
        "reconstructions": Path("./power_reconstructions"),
        "distributions": Path("./data_distributions"),
        "lasers": Path("./data_lasers")
    }

# 初始化目录
for d in Config.SAVE_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

class Decoder(nn.Module):
    """Decoder network architecture"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def preprocess_data(data: np.ndarray, indices: List[int]) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """数据预处理与归一化"""
    labels = data[:, 100:][:, indices]
    powers = data[:, :100]
    
    label_scaler = MinMaxScaler().fit(labels)
    power_scaler = MinMaxScaler().fit(powers)
    
    return labels, powers, label_scaler, power_scaler

def create_dataloader(powers: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> DataLoader:
    """创建PyTorch数据加载器"""
    power_tensor = torch.tensor(powers, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return DataLoader(TensorDataset(power_tensor, label_tensor), batch_size=batch_size, shuffle=True)

def train_decoder(
    decoder: nn.Module,
    data: np.ndarray,
    label_scaler: MinMaxScaler,
    power_scaler: MinMaxScaler,
    indices: List[int],
    iteration: int,
    num_epochs: int = 100
) -> None:
    """模型训练流程"""
    labels, powers = data[:, 100:][:, indices], data[:, :100]
    
    # 使用预训练的scaler进行转换
    norm_labels = label_scaler.transform(labels)
    norm_powers = power_scaler.transform(powers)
    
    dataloader = create_dataloader(norm_powers, norm_labels)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(decoder.parameters())
    
    # 模型检查点路径
    model_path = Config.SAVE_DIRS["models"] / f"decoder_{iteration}.pth"
    
    # # 加载已有模型
    # if model_path.exists():
    #     decoder.load_state_dict(torch.load(model_path, weights_only=True))
    #     print(f"Loaded existing model from {model_path}")
    #     return
    
    # 训练循环
    for epoch in range(num_epochs):
        decoder.train()
        for batch in dataloader:
            power_seq, labels = batch
            optimizer.zero_grad()
            reconstructed = decoder(labels)
            loss = criterion(reconstructed, power_seq)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 9:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")
    
    torch.save(decoder.state_dict(), model_path)

def generate_power_sequences(
    decoder: nn.Module,
    label_scaler: MinMaxScaler,
    power_scaler: MinMaxScaler,
    stats: Tuple[np.ndarray, np.ndarray],
    indices: List[int],
    num_samples: int = 5
) -> np.ndarray:
    """生成功率序列"""
    mean, std = stats
    sample_ranges = [np.linspace(m-0.2*s, m+0.2*s, num_samples) for m, s in zip(mean, std)]
    
    combinations = label_scaler.transform(list(itertools.product(*sample_ranges)))
    with torch.no_grad():
        generated = decoder(torch.tensor(combinations, dtype=torch.float32))
    
    generated = power_scaler.inverse_transform(generated.numpy())
    return np.clip(generated, 0, None)

def visualize_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    iteration: int,
    prefix: str = "power"
) -> None:
    """可视化对比"""
    plt.figure(figsize=(12, 6))
    plt.plot(original, label="Original", color="blue")
    plt.plot(reconstructed, label="Reconstructed", color="orange", linestyle="--")
    plt.title(f"{prefix.capitalize()} Comparison (Iteration {iteration})")
    plt.xlabel("Time Step" if prefix == "power" else "Value")
    plt.ylabel(prefix.capitalize())
    plt.legend()
    
    save_path = Config.SAVE_DIRS["reconstructions"] / f"{prefix}_comparison_{iteration}.png"
    plt.savefig(save_path)
    plt.close()

def plot_distribution(data: np.ndarray, iteration: int, index_vector: List[int]) -> None:
    """绘制标签分布直方图 (原plot_distribution)"""
    features = data[:, 100:][:, index_vector]
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    n_features = len(index_vector)

    fig_width = min(20, 5 * n_features)
    plt.figure(figsize=(fig_width, 5))
    
    for i in range(n_features):
        ax = plt.subplot(1, n_features, i+1)
        ax.set_title(f"Label {index_vector[i]}\nμ={means[i]:.2f}\nσ={stds[i]:.2f}", 
                    fontsize=8, pad=2)
        ax.hist(features[:, i], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.tick_params(axis='both', labelsize=6)
    
    # 手动调整布局
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        wspace=0.3,
        top=0.8
    )
    
    save_path = Config.SAVE_DIRS["distributions"] / f"data_distribution_{iteration}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_all_laser(data: np.ndarray, iteration: int) -> None:
    """绘制所有激光时序图 (原plot_all_laser)"""
    laser = data[:, :100]
    time = np.linspace(0, 5.69, 100)  # 更精确的时间生成方式
    
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
    
    save_path = Config.SAVE_DIRS["lasers"] / f"data_laser_{iteration}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def process_task(index: int, new_dir: str, Laser_grid: np.ndarray) -> None:
    """直接使用进程池处理单个任务"""
    pm.generate_input_data1D(new_dir, index, Laser_grid[index])
    pm.run_command_1D(new_dir, index)

def save_npy(program_name,old_data,num_samples):
    inp_data=[]
    fit_data=[]
    for i in range(num_samples):
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

def main_workflow():
    # 初始化数据
    data = np.load(Config.DATA_PATHS["input"])
    labels, powers, label_scaler, power_scaler = preprocess_data(data, Config.OUTPUT_INDICES)
    
    # 初始可视化
    plot_distribution(data, iteration=-1, index_vector=Config.OUTPUT_INDICES)
    plot_all_laser(data, iteration=-1)
    
    # 初始化模型
    decoder = Decoder(len(Config.OUTPUT_INDICES), Config.HIDDEN_SIZE, Config.INPUT_SIZE)
    
    # 主训练循环
    for iteration in range(Config.NUM_ITERATIONS):  # 示例运行3个迭代
        print(f"\n=== Iteration {iteration+1}/{Config.NUM_ITERATIONS} Training===")
        
        # 模型训练
        train_decoder(decoder, data, label_scaler, power_scaler, Config.OUTPUT_INDICES, iteration)
        
        # 功率重建可视化
        sample_labels = label_scaler.transform(labels[:1])
        with torch.no_grad():
            reconstructed = decoder(torch.tensor(sample_labels, dtype=torch.float32))
        
        # original_power = power_scaler.inverse_transform(powers[:1])
        reconstructed_power = power_scaler.inverse_transform(reconstructed.numpy())
        visualize_comparison(powers[:1].flatten(), reconstructed_power.flatten(), iteration)
        
        # 生成新序列
        stats = (labels.mean(axis=0), labels.std(axis=0))
        Laser_grid = generate_power_sequences(decoder, label_scaler, power_scaler, stats, Config.OUTPUT_INDICES, Config.NUM_SAMPLES)

        # 添加原始列生成逻辑
        new_column = np.full((Laser_grid.shape[0], 100), 0.05747)
        new_column[:, 0] = 0
        Laser_grid = np.hstack((new_column, Laser_grid))

        # 主流程中修改为
        print(f"\n=== Iteration {iteration+1}/{Config.NUM_ITERATIONS} Generated===")
        with ProcessPoolExecutor(max_workers=60) as pool:
            futures = [pool.submit(process_task, i, Config.NEW_DIR, Laser_grid) 
                    for i in range(Laser_grid.shape[0])]

        # 保持原始数据保存方式
        num_samples=len(Config.OUTPUT_INDICES)**Config.NUM_SAMPLES
        data, new_data = save_npy(Config.program_name,data, num_samples)
        
        # 更新可视化
        plot_distribution(new_data, iteration=iteration, index_vector=Config.OUTPUT_INDICES)
        plot_all_laser(new_data, iteration=iteration)
    # 最终保存
    np.save(Config.DATA_PATHS["output"], data)
    print("Workflow completed successfully!")

if __name__ == "__main__":
    main_workflow()