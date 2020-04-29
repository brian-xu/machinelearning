# import matplotlib; matplotlib.use("TkAgg")  # Uncomment to display animation on PyCharm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cost(theta: np.array, x: np.array, y: np.array, reg: float = 0) -> (float, np.array):
    """
    Determine cost with the current theta using squared-difference loss.
    theta: n x 1 vector
    x: m x n vector
    y: m x 1 vector
    J: scalar
    grad: n x 1 vector
    """
    m, n = x.shape
    h_x = x @ theta.T
    diff = y - h_x

    squared_diff = np.power(diff, 2)
    cost_reg = theta[1:] ** 2
    J = np.sum(squared_diff / (2 * m)) + reg * np.sum(cost_reg)

    grad = (h_x - y).T @ x / m

    return J, grad


def gradient_descent(theta: np.array, x: np.array, y: np.array, alpha: int, iters: int, reg: float = 0) -> (np.array,):
    """
    Perform gradient descent with learning rate alpha for iters iterations.
    theta: n x 1 vector
    x: m x n vector
    y: m x 1 vector
    alpha: scalar
    iters: scalar
    reg: scalar
    """
    m, n = x.shape
    J_iter = np.zeros((iters, 1))  # Store cost history
    theta_iter = np.zeros((iters, n))  # Store theta history
    for i in range(0, iters):
        J_iter[i], grad = cost(theta, x, y)
        theta_iter[i, :] = theta
        reg_adj = np.array([1] + [1 - (alpha * reg) / m for _ in range(n - 1)])
        reg_adj = reg_adj.reshape(theta.shape)
        theta = (theta * reg_adj) - grad * alpha
    return theta[0], J_iter, theta_iter


def regularize(x: np.array) -> (np.array,):
    """
    Regularize input array x with mean 0 and standard deviation 1.
    x: m x n vector
    """
    m, n = x.shape
    mean = np.zeros((m, 1))
    std = np.ones((m, 1))
    for col in range(1, n):
        mean[col] = np.mean(x[:, col])
        std[col] = np.std(x[:, col])
        x[:, col] = (x[:, col] - mean[col]) / std[col]
    return x, mean, std


with open("train.csv") as f:
    train = pd.read_csv(f)
    train = train.to_numpy()
with open("test.csv") as f:
    test = pd.read_csv(f)
    test = test.to_numpy()

m, n = train.shape

x = np.hstack((np.ones((m, 1)), train[:, 0:n - 1]))
y = train[:, n - 1].reshape((m, 1))

mean = np.zeros((m, 1))
std = np.ones((m, 1))
x, mean, std = regularize(x)

theta = np.zeros((1, n))
theta, J_iter, theta_iter = gradient_descent(theta, x, y, 0.01, 1500)

# Below code is not vectorized but it's just visualization code
total = 0
predicted = 0
for t in test:
    predicted += t[1] ** 2
    total += (theta[1] * (t[0] - mean[1]) / std[1] + theta[0]) ** 2

print("Test set accuracy:", total * 100 / predicted)

fig, ax = plt.subplots()

x_vals = np.linspace(0, 100)


def animate(i):
    p_x = theta_iter[i][1] * (x_vals - mean[1]) / std[1] + theta_iter[i][0]
    ax.clear()
    ax.scatter(test[:, 0], test[:, 1]),
    regression_line, = ax.plot(x_vals, p_x, 'black', linewidth=1)
    regression_line.set_label(f'Cost: {J_iter[i]}')
    ax.legend()


ani = animation.FuncAnimation(fig, animate, frames=len(theta_iter), interval=1, repeat=False)
animate(-1)
plt.show()
