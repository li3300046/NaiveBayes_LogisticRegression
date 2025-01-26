from scipy import io
import numpy as np

# ���� mnist_data.mat �ļ�
file_path = 'mnist_data.mat'
data = io.loadmat(file_path)

# ��ӡ�ļ������б����ļ�
print("Keys in the loaded .mat file:")
print(data.keys())

# ��ȡѵ���Ͳ�������
trX = data.get('trX')  # ѵ����
trY = data.get('trY')  # ѵ����ǩ
tsX = data.get('tsX')  # ���Լ�
tsY = data.get('tsY')  # ���Ա�ǩ

# ���������״
print("\nShapes of the datasets:")
print(f"trX shape: {trX.shape}")
print(f"trY shape: {trY.shape}")
print(f"tsX shape: {tsX.shape}")
print(f"tsY shape: {tsY.shape}")

# ��trX�ֳ��������ݣ� һ�������trY = 0�� ��һ�������trY = 1,����ӡ
group_0_mask = (trY == 0)  # ѵ����ǩΪ 0 ������
group_1_mask = (trY == 1)  # ѵ����ǩΪ 1 ������

group_0_trX = trX[group_0_mask.flatten()]  # ��ȡ��ǩΪ 0 ������

group_1_trX = trX[group_1_mask.flatten()]  # ��ȡ��ǩΪ 1 ������

# ��ӡ������������״
print("\nShapes of the two groups:")
print(f"Group with trY = 0: {group_0_trX}")
print(f"Group with trY = 1: {group_1_trX}")

# �����������ݵľ�ֵ��Э�������
mean_group_0 = np.mean(group_0_trX, axis=0)  # ���� trY = 0 �ľ�ֵ
cov_group_0 = np.cov(group_0_trX, rowvar=False)  # ���� trY = 0 ��Э�������

mean_group_1 = np.mean(group_1_trX, axis=0)  # ���� trY = 1 �ľ�ֵ
cov_group_1 = np.cov(group_1_trX, rowvar=False)  # ���� trY = 1 ��Э�������

# ��ӡ��ֵ��Э�������
print("\nMean and Covariance of Group with trY = 0:")
print(f"Mean: {mean_group_0}")
print(f"Covariance Matrix: {cov_group_0}")

print("\nMean and Covariance of Group with trY = 1:")
print(f"Mean: {mean_group_1}")
print(f"Covariance Matrix: {cov_group_1}")