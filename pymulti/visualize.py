import numpy as np
import matplotlib.pyplot as plt
from .process_1D import DataProcessor
from scipy.stats import gaussian_kde


def plot_shell(fort_path, ax=None,type=None):
    processor = DataProcessor()
    processor.read_fort10_file(fort_path)
    output = processor.extract(['X', 'TIME', "LPOWER"])


    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if type==None:
        for i in range(output["X"].shape[0]):
            ax.plot(output["TIME"].reshape(-1), output["X"][i, :], color='blue', linewidth=0.5)
    elif type=='high':
        for i in range(output["X"].shape[0]):
            if i < 181 and i > 61:
                ax.plot(output["TIME"].reshape(-1), output["X"][i, :], color='blue', linewidth=0.5)
    ax.plot(output["TIME"].reshape(-1), output["X"][0, :], color='blue', linewidth=0.5, label='Shell')
    ax.plot(output["TIME"].reshape(-1), output["LPOWER"].reshape(-1) / 2e21, label='Laser Power', color='red')
    ax.set_ylim([0, 0.13])
    ax.set_xlim([0, 0.7e-8])
    ax.legend()

    return fig

def plot_vimplo(fort_path, ax=None):
    processor = DataProcessor()
    processor.read_fort10_file(fort_path)
    output = processor.extract(['TIME', "VIMPLO"])

    # 如果没有传入 ax，则创建一个新的
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()  # 获取当前轴所在的图形对象

    for i in range(output["VIMPLO"].shape[0]):
        ax.plot(output["TIME"].reshape(-1), output["VIMPLO"][i, :])
    ax.plot(output["TIME"].reshape(-1), output["VIMPLO"].reshape(-1), label='VIMPLO')  # 添加图例标签
    ax.legend()

    return fig

def plot_heatmap(npy_data,xlabel='x',ylabel='y'):
    fig, ax = plt.subplots()
    h = ax.hist2d(npy_data[:, 0], npy_data[:, 1], bins=20, edgecolor='black')
    ax.figure.colorbar(h[3], ax=ax)
    ax.set_title('2D Histogram')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def plot_contour(npy_data,xlabel='x',ylabel='y'):
    x = npy_data[:,0]
    y = npy_data[:,1]

    xy = np.vstack([x, y])

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    zi = gaussian_kde(xy)(np.vstack([X.ravel(), Y.ravel()]))
    zi = zi.reshape(X.shape)

    plt.contour(X, Y, zi, levels=14, cmap='viridis')
    plt.colorbar()
    plt.scatter(x, y, color='r')
    plt.title('Density Contour Plot with Scatter Points')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
