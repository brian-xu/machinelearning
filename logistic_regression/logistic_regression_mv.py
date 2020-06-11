# import matplotlib; matplotlib.use("TkAgg")  # Uncomment to display animation on PyCharm
import itertools

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from scipy.spatial.qhull import ConvexHull

from import_datasets import download_wine

sns.set()

download_wine()

with open("wine.data") as f:
    wine = pd.read_csv(f)
    headers = wine.columns

wine['Alcohol'] = (wine['Alcohol'] == 1).astype(int)

data = wine.to_numpy()

sns.pairplot(wine, hue='Alcohol')
plt.show()

global theta_iter
theta_iter = []


def sigmoid(x: np.array) -> np.array:
    """
    Returns the sigmoid transformation of each value in an array.
    x: m x n array
    """
    return 1 / (1 + np.exp(-x))


def cost(theta: np.array, x: np.array, y: np.array, reg: float = 0) -> (float, np.array):
    """
    Determine the cost and gradient for the current theta using sigmoid loss.
    theta: 1 x n vector
    x: m x n vector
    y: m x 1 vector
    J: scalar
    grad: 1 x n vector
    """
    global theta_iter
    m, n = x.shape

    theta = theta.reshape((1, n))
    theta_iter.append(theta)

    h_x = sigmoid(x @ theta.T)
    pos = y * np.log(h_x)
    neg = (1 - y) * np.log(1 - h_x)

    log_error = -pos - neg
    J_reg = theta[1:] ** 2
    J = np.sum(log_error / m) + np.sum(J_reg * reg / (2 * m))

    grad = (h_x - y).T @ x
    grad_reg = np.append([0], theta[0, 1:] * (reg / m)).reshape(grad.shape)

    return J, grad + grad_reg


def map_features(x: np.array, n_features: int):
    """
    Create polynomial features using exponential combinations of input features.
    x: m x n array
    n_features: scalar determining the highest power each feature should be raised to.
    """
    m, n = x.shape
    feature_map = [range(0, n_features + 1)] * n
    poly_features = tuple(itertools.product(*feature_map))
    poly_x = np.hstack([np.prod(np.power(x, feature), axis=1).reshape(m, 1) for feature in poly_features])
    return poly_x


def regularize(x: np.array) -> (np.array,):
    """
    Regularize input array x with mean 0 and standard deviation 1.
    x: m x n vector
    """
    m, n = x.shape
    mean = np.zeros((1, n))
    std = np.ones((1, n))
    for col in range(0, n):
        mean[0, col] = np.mean(x[:, col])
        std[0, col] = np.std(x[:, col])
        x[:, col] = (x[:, col] - mean[0, col]) / std[0, col]
    return x, mean, std


params = [6, 13, 0]

data = data[:, params]
np.random.shuffle(data)
test_len = int(np.round(len(data) * 3 / 10))
train = data[:-test_len]
test = data[-test_len:]

x = train[:, 0:-1]
features = 4
x, mean, std = regularize(x)
x = map_features(x, features)

m, n = x.shape

y = train[:, -1].reshape((m, 1))

test_x = test[:, 0:-1]
test_x = (test_x - mean) / std
test_x = map_features(test_x, features)
test_y = test[:, -1].reshape((-1, 1))

theta = np.zeros((1, n))
theta = opt.fmin_tnc(func=cost, x0=theta, args=(x, y))[0]

theta_iter.append(theta.reshape((1, n)))

total = 0
correct = 0

for index, t in enumerate(test_x):
    actual = test_y[index, 0]
    predicted = sigmoid(t @ theta)
    if np.round(predicted) == actual:
        correct += 1
    total += 1

print("Test set accuracy:", correct * 100 / total)

# Below code is not vectorized but it's just visualization code

fig, ax = plt.subplots()


def decision_boundary(boundary_x: np.array, boundary_y: np.array, theta: np.array, n_features: int,
                      mean: np.array, std: np.array, max_bound: float = 0.1) -> np.array:
    """
    Find the decision boundary for multivariate logistic regression.
    graph_boundaries: array determining the smoothness of the boundary.
    theta: n x 1 array
    n_features: scalar determining the highest power each feature should be raised to.
    max_bound: scalar determining the largest loss that is included in the boundary.
    """
    z = []
    for i in boundary_x:
        for j in boundary_y:
            if 0 <= map_features((np.array([[i, j]]) - mean) / std, n_features) @ theta.T:
                z.append(np.array([i, j]))
    return np.array(z)


boundary_x = np.linspace(np.amin(test[:, 0]), np.amax(test[:, 0]), 50)
boundary_y = np.linspace(np.amin(test[:, 1]), np.amax(test[:, 1]), 50)
accepted = np.array([p for p in test if p[2] == 1])
rejected = np.array([p for p in test if p[2] == 0])

x1 = headers[params[0]]
x2 = headers[params[1]]


def animate(i):
    global theta_iter
    ax.clear()
    ax.scatter(accepted[:, 0], accepted[:, 1], c='dodgerblue')
    ax.scatter(rejected[:, 0], rejected[:, 1], c='firebrick')
    ax.legend(['Alcohol 1', 'Alcohol 2/3'], loc=0)
    ax.set_xlabel(x1)
    ax.set_ylabel(x2)
    ax.set_title(f'Alcohol Type Based On {x1} And {x2}')
    p_x = decision_boundary(boundary_x, boundary_y, theta_iter[i], features, mean, std)
    if len(p_x) > 2:
        hull = ConvexHull(p_x)
        for simplex in hull.simplices:
            ax.plot(p_x[simplex, 0], p_x[simplex, 1], 'k-')


ani = animation.FuncAnimation(fig, animate, frames=len(theta_iter), interval=75, repeat=False)
animate(-1)
plt.show()
