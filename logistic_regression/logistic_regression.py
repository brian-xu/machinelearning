# import matplotlib; matplotlib.use("TkAgg")  # Uncomment to display animation on PyCharm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns

from import_datasets import download_grades

sns.set()

download_grades()

with open("student-mat.csv") as f:
    student_grades = pd.read_csv(f, delimiter=';')
    headers = student_grades.columns

student_grades['G1'] = pd.to_numeric(student_grades['G1'])
student_grades['G2'] = pd.to_numeric(student_grades['G2'])
student_grades['G3'] = (student_grades['G3'] > 10).astype(int)

sns.relplot(x='G1', y='G2', hue='G3', data=student_grades)
plt.show()

data = student_grades.to_numpy()

global theta_iter
theta_iter = []


def add_ones(x: np.array) -> np.array:
    """
    Adds a column of ones to the given 2D array.
    x: m x n array.
    """
    m, n = x.shape
    return np.hstack((np.ones((m, 1)), x))


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

    grad = (h_x - y).T @ x / m
    grad_reg = np.append([0], theta[0, 1:] * (reg / m)).reshape(grad.shape)
    grad = grad + grad_reg

    return J, grad


def predict(theta: np.array, x: np.array) -> np.array:
    """
    Predict the output at the given x and theta.
    x: m x n array
    theta: 1 x n array
    """
    m, n = x.shape
    theta = theta.reshape((1, n))
    return np.round(sigmoid(x @ theta.T))


params = [-3, -2, -1]

data = data[:, params].astype('float64')
np.random.shuffle(data)
test_len = int(np.round(len(data) * 3 / 10))
train = data[:-test_len]
test = data[-test_len:]

m, n = train.shape

x = add_ones(train[:, 0:n - 1])
y = train[:, n - 1].reshape((m, 1))

theta = np.zeros((1, n))
theta = opt.fmin_tnc(func=cost, x0=theta, args=(x, y))[0]

theta_iter.append(theta.reshape((1, n)))

test_x = add_ones(test[:, :-1])
test_y = test[:, -1:]

p_x = predict(theta, test_x)

print("Test set accuracy:", np.sum(p_x == test_y) / len(test))

# Below code is not vectorized but it's just visualization code

fig, ax = plt.subplots()


def decision_boundary(x: np.array, theta: np.array) -> np.array:
    """
    Calculates the decision boundary line for a two-variable linear regression.
    x: m x 3 array
    theta: 1 x 3 array
    """
    theta = theta[0]
    if theta[2] == 0:
        return x * 0
    return -(theta[0] + theta[1] * x) / theta[2]


reg_x = np.arange(int(min(test[:, 0])), 20)
passing = np.array([p for p in test if p[2] == 1])
not_passing = np.array([p for p in test if p[2] == 0])

x1 = headers[params[0]]
x2 = headers[params[1]]


def animate(i):
    global theta_iter
    ax.clear()
    ax.scatter(passing[:, 0], passing[:, 1], c='blue')
    ax.scatter(not_passing[:, 0], not_passing[:, 1], c='red')
    p_x = decision_boundary(reg_x, theta_iter[i])
    ax.plot(reg_x, p_x, 'black')
    ax.legend([f'Cost: {cost(theta_iter[i], x, y)[0]}', 'Passed', 'Failed'], loc=3)
    ax.set_xlabel(x1)
    ax.set_ylabel(x2)
    ax.set_title('G3 Passing Status')


ani = animation.FuncAnimation(fig, animate, frames=len(theta_iter), interval=40, repeat=False)
animate(-1)
plt.show()
