# import matplotlib; matplotlib.use("TkAgg")  # Uncomment to display animation on PyCharm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

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


with open("train.csv") as f:
    train = pd.read_csv(f)
    train = train.to_numpy()
    test = train

m, n = train.shape

x = np.hstack((np.ones((m, 1)), train[:, 0:n - 1]))
y = train[:, n - 1].reshape((m, 1))

theta = np.zeros((1, n))
theta = opt.fmin_tnc(func=cost, x0=theta, args=(x, y))[0]

print(theta)

fig, ax = plt.subplots()


def decision_boundary(x: np.array, theta: np.array) -> np.array:
    theta = theta[0]
    if theta[2] == 0:
        return x * 0
    return -(theta[0] + theta[1] * x) / theta[2]


reg_x = np.arange(int(min(test[:, 0])), 100)
accepted = np.array([p for p in test if p[2] == 1])
rejected = np.array([p for p in test if p[2] == 0])


def animate(i):
    global theta_iter
    ax.clear()
    ax.scatter(accepted[:, 0], accepted[:, 1], c='blue')
    ax.scatter(rejected[:, 0], rejected[:, 1], c='red')
    ax.legend(['Accepted', 'Rejected'])
    p_x = decision_boundary(reg_x, theta_iter[i])
    ax.plot(reg_x, p_x, 'black')
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    ax.set_title('Accepted to College')


ani = animation.FuncAnimation(fig, animate, frames=len(theta_iter), interval=40, repeat=False)
animate(-1)
plt.show()

total = 0
correct = 0

for t in test:
    actual = t[-1]
    predicted = sigmoid(np.append([1], t[0:-1]) @ theta)
    if np.round(predicted) == actual:
        correct += 1
    total += 1

print("Test set accuracy:", correct * 100 / total)
