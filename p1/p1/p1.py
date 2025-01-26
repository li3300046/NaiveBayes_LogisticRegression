import numpy as np
from scipy.io import loadmat


def gaussian_log_pdf(x, mean, cov_inv, log_det_cov):
    diff = x - mean
    dim = mean.shape[0]
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    log_denominator = 0.5 * (dim * np.log(2 * np.pi) + log_det_cov)
    return exponent - log_denominator

file_path = "mnist_data.mat"
data = loadmat(file_path)
trX = data['trX']
trY = data['trY']
tsX = data['tsX']
tsY = data['tsY']

group_0_mask = (trY == 0).flatten()
group_1_mask = (trY == 1).flatten()
group_0_trX = trX[group_0_mask]
group_1_trX = trX[group_1_mask]

mean_group_0 = np.mean(group_0_trX, axis=0)
cov_group_0 = np.cov(group_0_trX, rowvar=False)
mean_group_1 = np.mean(group_1_trX, axis=0)
cov_group_1 = np.cov(group_1_trX, rowvar=False)

# 初始化正则化常数
epsilon = 1e-6

# 正则化协方差矩阵（避免奇异）
cov_group_0 += epsilon * np.eye(cov_group_0.shape[0])
cov_group_1 += epsilon * np.eye(cov_group_1.shape[0])

# 检查并处理行列式为零的情况
det_cov_0 = np.linalg.det(cov_group_0)
det_cov_1 = np.linalg.det(cov_group_1)
if np.isclose(det_cov_0, 0):
    print("Warning: Det(cov_group_0) is near zero.")
    det_cov_0 = epsilon  # 替代为一个小值

if np.isclose(det_cov_1, 0):
    print("Warning: Det(cov_group_1) is near zero.")
    det_cov_1 = epsilon  # 替代为一个小值

# 计算对数行列式
log_det_cov_0 = np.log(det_cov_0)
log_det_cov_1 = np.log(det_cov_1)

# 计算逆矩阵（或伪逆）
try:
    cov_inv_0 = np.linalg.inv(cov_group_0)
except np.linalg.LinAlgError:
    print("Warning: Using pseudo-inverse for cov_group_0")
    cov_inv_0 = np.linalg.pinv(cov_group_0)

try:
    cov_inv_1 = np.linalg.inv(cov_group_1)
except np.linalg.LinAlgError:
    print("Warning: Using pseudo-inverse for cov_group_1")
    cov_inv_1 = np.linalg.pinv(cov_group_1)


prior_0 = len(group_0_trX) / len(trX)
prior_1 = len(group_1_trX) / len(trX)

log_prior_0 = np.log(prior_0)
log_prior_1 = np.log(prior_1)

log_posterior_0 = log_prior_0 + gaussian_log_pdf(tsX, mean_group_0, cov_inv_0, log_det_cov_0)
log_posterior_1 = log_prior_1 + gaussian_log_pdf(tsX, mean_group_1, cov_inv_1, log_det_cov_1)

predictions = (log_posterior_1 > log_posterior_0).astype(int)

print("\nTrue labels (tsY):")
print(tsY.flatten())

print("\nPredicted labels:")
print(predictions)

accuracy = np.mean(predictions == tsY.flatten()) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
