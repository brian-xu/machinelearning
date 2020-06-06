# import matplotlib; matplotlib.use("TkAgg")  # Uncomment to display animation on PyCharm
import itertools

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.spatial import ConvexHull

style.use('ggplot')

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
    poly_x = np.hstack([np.power(x, feature) for feature in poly_features])
    return poly_x


with open("train_mv.csv") as f:
    train = pd.read_csv(f)
    train = train.to_numpy()
    test = train

x = train[:, 0:-1]
features = 6
x = map_features(x, features)

m, n = x.shape

y = train[:, -1].reshape((m, 1))

theta = np.zeros((1, n))
theta = opt.fmin_tnc(func=cost, x0=theta, args=(x, y))[0]

theta_iter.append(theta.reshape((1, n)))

total = 0
correct = 0

for index, t in enumerate(x):
    actual = y[index][-1]
    predicted = sigmoid(t @ theta)
    if np.round(predicted) == actual:
        correct += 1
    total += 1

print("Test set accuracy:", correct * 100 / total)

# Below code is not vectorized but it's just visualization code

fig, ax = plt.subplots()


def decision_boundary(graph_boundaries: np.array, theta: np.array, n_features: int, max_bound: float = 0.3) -> np.array:
    """
    Find the decision boundary for multivariate logistic regression.
    graph_boundaries: array determining the smoothness of the boundary.
    theta: n x 1 array
    n_features: scalar determining the highest power each feature should be raised to.
    max_bound: scalar determining the largest loss that is included in the boundary.
    """
    z = []
    for i in graph_boundaries:
        for j in graph_boundaries:
            if 0 <= map_features(np.array([[i, j]]), n_features) @ theta.T <= max_bound:
                z.append(np.array([i, j]))
    return np.array(z)


graph_boundaries = np.linspace(np.amin(test[:, 0:2]), np.amax(test[:, 0:2]), 50)
accepted = np.array([p for p in test if p[2] == 1])
rejected = np.array([p for p in test if p[2] == 0])


def animate(i):
    global theta_iter
    ax.clear()
    ax.scatter(accepted[:, 0], accepted[:, 1], c='dodgerblue')
    ax.scatter(rejected[:, 0], rejected[:, 1], c='firebrick')
    ax.legend(['Accepted', 'Rejected'], loc=1)
    p_x = decision_boundary(graph_boundaries, theta_iter[i], features)
    if len(p_x) > 2:
        hull = ConvexHull(p_x)
        for simplex in hull.simplices:
            ax.plot(p_x[simplex, 0], p_x[simplex, 1], 'k-')
    ax.set_xlabel('Test 1 Results')
    ax.set_ylabel('Test 2 Results')
    ax.set_title('Microchip Validation')


ani = animation.FuncAnimation(fig, animate, frames=len(theta_iter), interval=75, repeat=False)
animate(-1)
plt.show()
