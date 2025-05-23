import pymulti as pm
import numpy as np
import matplotlib.pyplot as plt

program_name='Multi-1D-Sample-LHS'
inp_data=[]
fit_data=[]
for i in range(60000):
    inp_data.append(pm.data1D_process_inp(program_name, i))
    fit_data.append(pm.data1D_process_fit(program_name, i))

data=np.concatenate((inp_data, fit_data), axis=1)

# 前24列的时间间隔和功率
time_power_part = data[:, :24]  # 前24列是时间区间和功率
other_columns = data[:, 24:]     # 后面k列的数据

n_samples = data.shape[0]

# 假设前12列是时间间隔（可能需要调整为你的时间间隔列）
# 后12列是对应的功率值
time_intervals = time_power_part[:, :12]  # 前12列为时间区间
power_values = time_power_part[:, 12:]    # 后12列为功率值
time_row = time_intervals[i].cumsum()

# 初始化一个空列表存储插值后的功率值
interp_powers = []

# 遍历每一行数据处理
for i in range(n_samples):
    power_row = power_values[i]   # 获取当前行的功率值

    # 假设时间区间是均匀分布的，或者根据实际时间生成新的时间点
    # 如果时间区间不是均匀分布，请调整以下代码中的时间点
    new_time = np.linspace(time_row[0], time_row[-1], 100)  # 生成100个等分的新时间点
    interp_power = np.interp(new_time, time_row, power_row)  # 插值得到新的功率值
    interp_powers.append(interp_power)

# 将插值后的功率数据转换为数组
interp_powers = np.array(interp_powers)

# 合并插值后的功率数据和后面k列
new_data = np.hstack((interp_powers, other_columns[:,3:]))

# 保存新的npy文件
np.save("LHS_data-0.8.npy", new_data)



def plot_all_laser(data):
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
    
    save_path = "data_laser-0.8.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

plot_all_laser(new_data)