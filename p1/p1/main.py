import numpy as np
from scipy.io import loadmat

def gaussian_log_pdf(x, mean, cov_inv, log_det_cov):
    diff = x - mean
    dim = mean.shape[0]
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    log_denominator = 0.5 * (dim * np.log(2 * np.pi) + log_det_cov)
    return exponent - log_denominator

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_gradient(X, y, theta):
    m = X.shape[0]
    predictions = sigmoid(X @ theta)
    error = y - predictions
    gradient = X.T @ error / m
    return gradient


def logistic_regression(X, y, alpha, num_iters, TX, TY):
    n_features = X.shape[1]
    theta = np.random.randn(n_features, 1) * 0.01

    TY = TY.flatten().astype(int)
    for i in range(num_iters):
        gradient = compute_gradient(X, y, theta)
        theta += alpha * gradient

        predictions_prob = sigmoid(TX @ theta)
        predictions = (predictions_prob >= 0.5).astype(int)

        predictions = np.array(predictions).flatten()
        TY = TY.flatten()

        # check current accuracy 
        accuracy = np.mean(predictions == TY) * 100 
        print(f"Iteration {i+1}/{num_iters}, Accuracy of Logistic Regression: {accuracy:.2f}%")

        print("Predictions (first 20):", predictions[:20].flatten())
        print("True labels (first 20):", TY[:20].flatten())

        print("Predictions (last 20):", predictions[-20:].flatten())
        print("True labels (last 20):", TY[-20:].flatten())
    return theta

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

epsilon = 1e-6

cov_group_0 += epsilon * np.eye(cov_group_0.shape[0])
cov_group_1 += epsilon * np.eye(cov_group_1.shape[0])

# if the det of cov == 0, use a epsilon to instead
det_cov_0 = np.linalg.det(cov_group_0)
det_cov_1 = np.linalg.det(cov_group_1)
if np.isclose(det_cov_0, 0):
    det_cov_0 = epsilon

if np.isclose(det_cov_1, 0):
    det_cov_1 = epsilon

log_det_cov_0 = np.log(det_cov_0)
log_det_cov_1 = np.log(det_cov_1)

# calclulate cov
try:
    cov_inv_0 = np.linalg.inv(cov_group_0)
except np.linalg.LinAlgError:
    cov_inv_0 = np.linalg.pinv(cov_group_0)

try:
    cov_inv_1 = np.linalg.inv(cov_group_1)
except np.linalg.LinAlgError:
    cov_inv_1 = np.linalg.pinv(cov_group_1)


prior_0 = len(group_0_trX) / len(trX)
prior_1 = len(group_1_trX) / len(trX)

log_prior_0 = np.log(prior_0)
log_prior_1 = np.log(prior_1)

log_posterior_0 = log_prior_0 + gaussian_log_pdf(tsX, mean_group_0, cov_inv_0, log_det_cov_0)
log_posterior_1 = log_prior_1 + gaussian_log_pdf(tsX, mean_group_1, cov_inv_1, log_det_cov_1)

predictions = (log_posterior_1 > log_posterior_0).astype(int)

# print("\nTrue labels (tsY):")
# print(tsY.flatten())

# print("\nPredicted labels:")
# print(predictions)

accuracy = np.mean(predictions == tsY.flatten()) * 100
print(f"\nNaive Bayes Accuracy: {accuracy:.2f}%")

trX = np.hstack((np.ones((trX.shape[0], 1)), trX))
tsX = np.hstack((np.ones((tsX.shape[0], 1)), tsX))

trY = trY.reshape(-1, 1)

# shuffle train data to make the model learn process smooth
shuffle_indices = np.random.permutation(trX.shape[0])
trX = trX[shuffle_indices]
trY = trY[shuffle_indices]

alpha = 0.1
num_iters = 1000
theta = logistic_regression(trX, trY, alpha, num_iters, tsX, tsY)
