import matplotlib.pyplot as plt
import numpy as np
import pandas


def cost(X: np.array, y: np.array, theta: np.array) -> float:
    """ X: m x n vector
        theta: n x 1 vector
        Y: m x 1 vector """
    m = np.shape(y)[0]
    h_x = X.dot(theta)
    diff = y - h_x
    J = np.power(diff, 2)
    return np.sum(J / (2 * m))


def gradient_descent(X: np.array, y: np.array, theta: np.array, alpha: int, num_iters: int) -> np.array:
    J_iter = np.zeros((num_iters, 1))  # Store cost history
    J_iter[0] = cost(X, y, theta)
    m = np.shape(y)[0]
    for iter in range(1, num_iters):
        h_x = X.dot(theta)
        diff = np.transpose(X).dot(h_x - y)
        theta_new = theta - (diff * alpha / m)
        J_iter[iter] = cost(X, y, theta_new)
        if J_iter[iter] < J_iter[iter - 1]:
            theta = theta_new  # Only update theta if cost decreases. This is irrelevant with a low enough learning
            # rate in linear regression, but can be important for more complex models.
    return theta


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
    train = train.to_numpy()
with open("test.csv") as f:
    test = pandas.read_csv(f)
    test = test.to_numpy()

m, n = train.shape

X = np.hstack((np.ones((m, 1)), train[:, 0:n - 1]))
X, mean, std = regularize(X, 1)
y = train[:, n - 1].reshape((700, 1))

theta = np.zeros((n, 1))

theta = gradient_descent(X, y, theta, 0.01, 1500)

# Below code is not vectorized but it's just visualization code

x = np.arange(100)
p_x = theta[1] * (x - mean[1]) / std[1] + theta[0]

print(theta)

plt.scatter(test[:, 0], test[:, 1])
plt.plot(x, p_x, 'g', linewidth=2)
plt.show()

total = 0
predicted = 0
for t in test:
    predicted += t[1] ** 2
    total += (theta[1] * (t[0] - mean[1]) / std[1] + theta[0]) ** 2

print("Test set accuracy:", total * 100 / predicted)
