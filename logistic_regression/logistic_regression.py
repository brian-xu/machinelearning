import numpy as np
import pandas
import scipy.optimize as opt


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def cost(theta: np.array, X: np.array, y: np.array, reg: float = 0) -> (float,):
    """ X: m x n vector
        theta: n x 1 vector
        Y: m x 1 vector """
    m = np.shape(y)[0]
    theta = theta.reshape((theta.shape[0], 1))
    h_x = sigmoid(X @ theta)
    pos = y * np.log(h_x)
    neg = (1 - y) * np.log(1 - h_x)

    J = -pos - neg
    J_reg = theta[1:] ** 2
    J = np.sum(J / m) + np.sum(J_reg * reg / (2 * m))

    grad = X.T @ (h_x - y)
    grad_reg = np.append([0], theta[1:] * (reg / m)).reshape(grad.shape)

    return J, grad + grad_reg


def regularize(X: np.array, start: int) -> (np.array,):
    m, n = X.shape
    mean = np.zeros((m, 1))
    std = np.ones((m, 1))
    for col in range(start, n):
        mean[col] = np.mean(X[:, col])
        std[col] = np.std(X[:, col])
        X[:, col] = (X[:, col] - mean[col]) / std[col]
    return X, mean, std


with open("train.csv") as f:
    train = pandas.read_csv(f)
    train.dropna(inplace=True)
    train = train.to_numpy()
    test = train

m, n = train.shape

X = np.hstack((np.ones((m, 1)), train[:, 0:n - 1]))
# X, mean, std = regularize(X, 1)
y = train[:, n - 1].reshape((m, 1))

theta = np.zeros((X.shape[1], 1))

theta = opt.fmin_tnc(func=cost, x0=theta, args=(X, y, 10))[0]

print(theta)

total = 0
correct = 0

for t in test:
    actual = t[-1]
    predicted = sigmoid(np.append([1], t[0:-1]) @ theta)
    if np.round(predicted) == actual:
        correct += 1
    total += 1

print("Test set accuracy:", correct * 100 / total)
