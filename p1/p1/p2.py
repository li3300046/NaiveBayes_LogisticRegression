import numpy as np
from scipy.io import loadmat


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

        accuracy = np.mean(predictions == TY) * 100 
        print(f"Iteration {i+1}/{num_iters}, Accuracy of Logistic Regression: {accuracy:.2f}%")

        print("Predictions (first 20):", predictions[:20].flatten())
        print("True labels (first 20):", TY[:20].flatten())

        print("Predictions (last 20):", predictions[-20:].flatten())  # 打印最后 20 个预测值
        print("True labels (last 20):", TY[-20:].flatten())          # 打印最后 20 个真实值
    return theta


file_path = "mnist_data.mat"
data = loadmat(file_path)
trX = data['trX']
trY = data['trY']
tsX = data['tsX']
tsY = data['tsY']

trX = np.hstack((np.ones((trX.shape[0], 1)), trX))
tsX = np.hstack((np.ones((tsX.shape[0], 1)), tsX))

trY = trY.reshape(-1, 1)

shuffle_indices = np.random.permutation(trX.shape[0])
trX = trX[shuffle_indices]
trY = trY[shuffle_indices]

alpha = 0.1
num_iters = 1000
theta = logistic_regression(trX, trY, alpha, num_iters, tsX, tsY)
