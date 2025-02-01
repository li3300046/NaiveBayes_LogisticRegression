import numpy as np
from scipy.io import loadmat

def gaussian_pdf(x, mean, std):
    std = np.where(std < 1e-1, 1e-1, std)  # max function here to avoid extreme value
    exponent = -0.5 * ((x - mean) / std) ** 2
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, theta):
    m = X.shape[0]
    predictions = sigmoid(X @ theta)
    error = y - predictions
    gradient = X.T @ error / m
    return gradient

def logistic_regression(X, y, alpha, num_iters):
    n_features = X.shape[1]
    theta = np.random.randn(n_features, 1) * 0.01

    for i in range(num_iters):
        # calculation
        gradient = compute_gradient(X, y, theta)
        theta += alpha * gradient
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

# feature extraction for digital 7
mean_group_0 = np.mean(group_0_trX, axis=0)
std_group_0 = np.std(group_0_trX, axis=0, ddof=1)

# feature extraction for digital 8
mean_group_1 = np.mean(group_1_trX, axis=0)
std_group_1 = np.std(group_1_trX, axis=0, ddof=1)

prior_0 = len(group_0_trX) / len(trX)
prior_1 = len(group_1_trX) / len(trX)

log_prior_0 = np.log(prior_0)
log_prior_1 = np.log(prior_1)

log_posterior_0 = log_prior_0 + np.sum(np.log(gaussian_pdf(tsX, mean_group_0, std_group_0)), axis=1)
log_posterior_1 = log_prior_1 + np.sum(np.log(gaussian_pdf(tsX, mean_group_1, std_group_1)), axis=1)


predictions = (log_posterior_1 > log_posterior_0).astype(int)
accuracy = np.mean(predictions == tsY.flatten()) * 100
print(f"Accuracy of Naive Bayes: {accuracy:.6f}%")

trX = np.hstack((np.ones((trX.shape[0], 1)), trX))
tsX = np.hstack((np.ones((tsX.shape[0], 1)), tsX))

trY = trY.reshape(-1, 1)

# shuffle train data to make the model learn process smooth
shuffle_indices = np.random.permutation(trX.shape[0])
trX = trX[shuffle_indices]
trY = trY[shuffle_indices]

alpha = 0.1
num_iters = 1000
theta = logistic_regression(trX, trY, alpha, num_iters)

# check current accuracy 
predictions_prob = sigmoid(tsX @ theta)
predictions = (predictions_prob >= 0.5).astype(int)
predictions = np.array(predictions).flatten()
TY = tsY.flatten()
accuracy = np.mean(predictions == tsY) * 100 
print(f"Accuracy of Logistic Regression: {accuracy:.6f}%")
