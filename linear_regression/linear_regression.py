# import matplotlib; matplotlib.use("TkAgg")  # Uncomment to display animation on PyCharm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cost(theta: np.array, x: np.array, y: np.array, reg: float = 0) -> (float, np.array):
    """
    Determine cost with the current theta using squared-difference loss.
    theta: 1 x n vector
    x: m x n vector
    y: m x 1 vector
    J: scalar
    grad: 1 x n vector
    """
    m, n = x.shape
    h_x = x @ theta.T
    diff = y - h_x

    squared_diff = np.power(diff, 2)
    J_reg = theta[1:] ** 2
    J = np.sum(squared_diff / (2 * m)) + reg * np.sum(J_reg)

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
        reg_adj = np.array([[1] + [1 - (alpha * reg) / m for _ in range(n - 1)]])
        theta = (theta * reg_adj) - grad * alpha
    return theta[0], J_iter, theta_iter


def regularize(x: np.array) -> (np.array,):
    """
    Regularize input array x with mean 0 and standard deviation 1.
    x: m x n vector
    """
    m, n = x.shape
    mean = np.zeros((1, n))
    std = np.ones((1, n))
    for col in range(1, n):
        mean[0, col] = np.mean(x[:, col])
        std[0, col] = np.std(x[:, col])
        x[:, col] = (x[:, col] - mean[0, col]) / std[0, col]
    return x, mean, std


params = [3, -1]

with open("retail.csv") as f:
    data = pd.read_csv(f)
    data = data.to_numpy()
    data = data[:, params]
    np.random.shuffle(data)
    test_len = int(np.round(len(data) * 3 / 10))
    train = data[:-test_len]
    test = data[-test_len:]

m, n = train.shape

x = np.hstack((np.ones((m, 1)), train[:, 0:n - 1]))
y = train[:, n - 1].reshape((m, 1))

mean = np.zeros((1, n))
std = np.ones((1, n))
x, mean, std = regularize(x)

theta = np.zeros((1, n))
theta, J_iter, theta_iter = gradient_descent(theta, x, y, 0.01, 1500)
theta = theta.reshape((1, n))

test_x = np.hstack((np.ones((test_len, 1)), test[:, 0:n - 1]))
test_y = test[:, n - 1].reshape((test_len, 1))

residual = np.sum((test_y - (((test_x - mean) / std) @ theta.T)) ** 2)
total = np.sum((test_y - np.mean(test_y)) ** 2)

print(f"R-squared for test set: {1 - (residual / total)}")

# Below code is not vectorized but it's just visualization code

fig, ax = plt.subplots()

x_vals = np.linspace(0, int(np.max(test_x[:, -1:]))).reshape((50, 1))

ax.set(ylim=(0, np.ceil(np.max(test[:, 1]) / 10) * 10))


def animate(i):
    ax.clear()
    ax.scatter(test[:, 0], test[:, 1])
    p_x = (theta_iter[i] @ ((np.hstack((np.ones((50, 1)), x_vals)) - mean) / std).T)
    regression_line, = ax.plot(x_vals, p_x, 'black', linewidth=1)
    regression_line.set_label(f'Cost: {J_iter[i]}')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    ax.set_title('House price per unit area')


ani = animation.FuncAnimation(fig, animate, frames=len(theta_iter), interval=1, repeat=False)
animate(-1)
plt.show()
