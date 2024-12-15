import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from optim import Adadelta, Adagrad, Adam, Adamax, AdamW, NAdam, RMSProp, RProp, SGD, SparseAdam

# 定义二维损失函数
def loss_function(x, y):
    return 3 * x ** 2 + 4 * y ** 2

# 计算损失函数关于x和y的偏导数（针对单个样本的情况）
def gradient(x, y):
    grad_x = 6 * x
    grad_y = 8 * y
    return np.array([grad_x, grad_y])

def train(optimizer_name, num_iterations=500):
    # 初始参数
    params = np.array([4.5, 4.5])
    x, y = params[0], params[1]
    # 创建优化器实例
    if optimizer_name == "Adadelta":
        optimizer = Adadelta()
    elif optimizer_name == "Adagrad":
        optimizer = Adagrad()
    elif optimizer_name == "Adam":
        optimizer = Adam()
    elif optimizer_name == "Adamax":
        optimizer = Adamax()
    elif optimizer_name == "AdamW":
        optimizer = AdamW()
    elif optimizer_name == "NAdam":
        optimizer = NAdam()
    elif optimizer_name == "RMSProp":
        optimizer = RMSProp()
    elif optimizer_name == "RProp":
        optimizer = RProp()
    elif optimizer_name == "SGD":
        optimizer = SGD()
    elif optimizer_name == "SparseAdam":
        optimizer = SparseAdam()
    else:
        raise Exception("Not Support This Optimizer!")

    # 用于存储每次迭代的x、y和损失值，方便可视化
    x_history = []
    y_history = []
    loss_history = []

    for _ in range(num_iterations):
        grads = gradient(x, y)
        params = np.array([x, y])
        updated_params = optimizer.update(params, grads)
        x, y = updated_params[0], updated_params[1]
        loss = loss_function(x, y)
        x_history.append(x)
        y_history.append(y)
        loss_history.append(loss)

    # 创建3D图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 生成用于绘制损失函数曲面的网格数据
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_function(X, Y)

    # 绘制损失函数曲面
    ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis')

    # 绘制参数x、y变化以及对应损失函数值的散点图
    ax.plot(x_history, y_history, loss_history, c='r', marker='.', label=f'{optimizer_name} Optimization Path')

    # 设置坐标轴标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Loss')

    # 设置图形标题
    ax.set_title(f'Parameter and Loss Change with {optimizer_name}')

    # 添加图例
    ax.legend()

    # 显示图形
    plt.savefig(f'result\\{optimizer_name}.png')
    plt.close()

    # 可视化损失下降过程
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss Decrease with {optimizer_name}')
    # 显示图形
    plt.savefig(f'result\\{optimizer_name}_loss.png')
    plt.close()

if __name__ == "__main__":
    names = ["Adadelta", "Adagrad", "Adam", "Adamax", "AdamW", "NAdam", "RMSProp", "RProp", "SGD", "SparseAdam"]
    for name in names:
        train(optimizer_name=name)