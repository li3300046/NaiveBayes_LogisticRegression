from scipy import io
import numpy as np

# 加载 mnist_data.mat 文件
file_path = 'mnist_data.mat'
data = io.loadmat(file_path)

# 打印文件中所有变量的键
print("Keys in the loaded .mat file:")
print(data.keys())

# 提取训练和测试数据
trX = data.get('trX')  # 训练集
trY = data.get('trY')  # 训练标签
tsX = data.get('tsX')  # 测试集
tsY = data.get('tsY')  # 测试标签

# 检查数据形状
print("\nShapes of the datasets:")
print(f"trX shape: {trX.shape}")
print(f"trY shape: {trY.shape}")
print(f"tsX shape: {tsX.shape}")
print(f"tsY shape: {tsY.shape}")

# 将trX分成两组数据， 一组仅包含trY = 0， 另一组仅包含trY = 1,并打印
group_0_mask = (trY == 0)  # 训练标签为 0 的样本
group_1_mask = (trY == 1)  # 训练标签为 1 的样本

group_0_trX = trX[group_0_mask.flatten()]  # 获取标签为 0 的样本

group_1_trX = trX[group_1_mask.flatten()]  # 获取标签为 1 的样本

# 打印分组后的数据形状
print("\nShapes of the two groups:")
print(f"Group with trY = 0: {group_0_trX}")
print(f"Group with trY = 1: {group_1_trX}")

# 计算两组数据的均值和协方差矩阵
mean_group_0 = np.mean(group_0_trX, axis=0)  # 计算 trY = 0 的均值
cov_group_0 = np.cov(group_0_trX, rowvar=False)  # 计算 trY = 0 的协方差矩阵

mean_group_1 = np.mean(group_1_trX, axis=0)  # 计算 trY = 1 的均值
cov_group_1 = np.cov(group_1_trX, rowvar=False)  # 计算 trY = 1 的协方差矩阵

# 打印均值和协方差矩阵
print("\nMean and Covariance of Group with trY = 0:")
print(f"Mean: {mean_group_0}")
print(f"Covariance Matrix: {cov_group_0}")

print("\nMean and Covariance of Group with trY = 1:")
print(f"Mean: {mean_group_1}")
print(f"Covariance Matrix: {cov_group_1}")