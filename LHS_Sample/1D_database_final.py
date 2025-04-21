from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pymulti as pm
from functools import partial
from pyDOE import lhs

# ================= 常量定义 =================
PROGRAM_NAME = 'Multi-1D-Sample-LHS'
N_SAMPLES = 60000
MAX_WORKERS_PROCESS = 60
MAX_WORKERS_THREAD = 4
MIN_DIFF_THRESHOLD = 1

# 激光参数（时间+功率）
LASER_TIME = [0.000000, 0.380000, 0.510000, 0.240000, 0.250000, 0.320000, 0.410000, 0.260000, 0.360000, 0.62, 2.21, 0.1300]
LASER_POWER =  [0.000000, 4.490000, 0.000000, 0.00, 3.480000, 4.060000, 5.010000, 5.880000, 13.47000, 28.0, 0.000000]
LASER_POWER1=[i*0.8 for i in LASER_POWER]
LASER_POWER_MIN = [i * 1.2 for i in LASER_POWER1]
LASER_POWER_MAX = [i * 0.8 for i in LASER_POWER1]

# ================= 派生参数 =================
n_time_steps = len(LASER_TIME)
total_time = sum(LASER_TIME)
time_min = np.array(LASER_TIME) * (1 - 0.2)  # TIME_RANGE 硬编码为0.2
time_max = np.array(LASER_TIME) * (1 + 0.2)
dimensions = n_time_steps + len(LASER_POWER)

# ================= 简化的采样函数 =================
def generate_samples(dimensions, n_samples):
    """直接生成时间和激光功率样本"""
    samples = lhs(dimensions, samples=n_samples)
    
    # 处理时间维度
    time_samples = []
    for i in range(n_samples):
        # 使用Dirichlet分布生成归一化时间
        normalized_time = np.random.dirichlet(np.ones(n_time_steps))
        adjusted_time = time_min + normalized_time * (time_max - time_min)
        adjusted_time *= total_time / adjusted_time.sum()  # 保持总时间不变
        time_samples.append(adjusted_time)
    
    samples[:, :n_time_steps] = np.array(time_samples)
    
    # 处理功率维度
    for col_idx in range(len(LASER_POWER)):
        col = n_time_steps + col_idx
        # 归一化到功率范围
        samples[:, col] = samples[:, col] * (LASER_POWER_MAX[col_idx] - LASER_POWER_MIN[col_idx]) + LASER_POWER_MIN[col_idx]
    return samples

# ================= 任务函数 =================
def process_task(index, new_dir, laser_params):
    """处理单个样本的任务，使用固定厚度"""
    pm.generate_input_data1D(
        new_dir, 
        index,
        laser_params,  # 只包含时间和功率参数
    )
    pm.run_command_1D(new_dir, index)

def thread_task(index, new_dir, laser_grid):
    """线程级任务分发"""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_THREAD) as executor:
        executor.submit(process_task, index, new_dir, laser_grid[index])

# ================= 主函数 =================
def main():
    # 生成样本
    laser_grid = generate_samples(dimensions, N_SAMPLES)
    Laser_grid=np.zeros((laser_grid.shape[0],laser_grid.shape[1]+1))
    Laser_grid[:,:-1]=laser_grid
    Laser_grid[:,-2]=Laser_grid[:,-3]
    print(Laser_grid.shape)
    print(Laser_grid[0])
    
    # 初始化工作目录并并行执行
    new_dir = pm.init1D(PROGRAM_NAME)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_PROCESS) as pool:
        list(pool.map(
            partial(thread_task, new_dir=new_dir, laser_grid=Laser_grid),
            range(len(Laser_grid))
        ))

if __name__ == '__main__':
    main()