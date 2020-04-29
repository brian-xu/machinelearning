import matplotlib;

matplotlib.use("TkAgg")  # Uncomment to display animation on PyCharm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.spatial import ConvexHull
import itertools

global theta_iter
theta_iter = []


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))


def map_features(x: np.array, n_features: int):
    m, n = x.shape
    poly_x = np.ones((m, 1))
    feature_map = [range(1, n_features + 1)] * n
    poly_features = tuple(itertools.product(*feature_map))
    for feature in poly_features:
        poly_x = np.hstack((poly_x, np.power(x, feature)))
    return poly_x


def cost(theta: np.array, x: np.array, y: np.array, reg: float = 0) -> (float, np.array):
    """
    Determine the cost and gradient for the current theta using sigmoid loss.
    theta: 1 x n vector
    x: m x n vector
    y: m x 1 vector
    J: scalar
    grad: n x 1 vector
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

fig, ax = plt.subplots()


def decision_boundary(graph_boundaries: np.array, theta: np.array, features: int) -> np.array:
    z = []
    for i in graph_boundaries:
        for j in graph_boundaries:
            if 0 <= map_features(np.array([[i, j]]), features) @ theta.T <= 0.3:
                z.append(np.array([i, j]))
    return np.array(z)


graph_boundaries = np.linspace(np.amin(test[:, 0:2]), np.amax(test[:, 0:2]), 100)
accepted = np.array([p for p in test if p[2] == 1])
rejected = np.array([p for p in test if p[2] == 0])


def animate(i):
    global theta_iter
    ax.clear()
    ax.scatter(accepted[:, 0], accepted[:, 1], c='dodgerblue')
    ax.scatter(rejected[:, 0], rejected[:, 1], c='firebrick')
    ax.legend(['Accepted', 'Rejected'], loc = 1)
    p_x = decision_boundary(graph_boundaries, theta_iter[i], features)
    if len(p_x) > 2:
        hull = ConvexHull(p_x)
        for simplex in hull.simplices:
            ax.plot(p_x[simplex, 0], p_x[simplex, 1], 'k-')
    ax.set_xlabel('Test 1 Results')
    ax.set_ylabel('Test 2 Results')
    ax.set_title('Microchip Validation')


ani = animation.FuncAnimation(fig, animate, frames=137, interval=75, repeat=False)
animate(-1)
plt.show()

total = 0
correct = 0

for index, t in enumerate(x):
    actual = y[index][-1]
    predicted = sigmoid(t @ theta)
    if np.round(predicted) == actual:
        correct += 1
    total += 1

print("Test set accuracy:", correct * 100 / total)
