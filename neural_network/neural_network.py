import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.optimize as opt


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x: np.array) -> np.array:
    return sigmoid(x) * (1 - sigmoid(x))


def predict(theta: np.array, layer_sizes: np.array, X: np.array) -> np.array:
    layers = []
    for layer in layer_sizes:
        layer_size = layer[0] * layer[1]
        layers.append(theta[:layer_size].reshape(layer))
        theta = theta[layer_size:]

    for i in range(len(layer_sizes)):
        m, n = X.shape
        h = np.hstack((np.ones((m, 1)), X))
        X = sigmoid(h @ layers[i].T)

    return np.argmax(X, axis=1)


def cost(theta: np.array, layer_sizes: np.array, labels_end: int,
         X: np.array, y: np.array, reg: float = 0, labels_start: int = 0) -> (float, np.array):
    """
    theta: unrolled array containing all layers
    layer_sizes: array indicating the size of layers
    labels_end: integer indicating the largest label in y
    X: m x n input layer
    y: m x 1 output with values ranging from 0-labels
    reg: regularization parameter
    labels_start: optional argument indicating the smallest label in y
    """
    m, n = X.shape
    layers = []
    a = []
    z = []
    grads = []
    y_i = np.zeros((m, labels_end - labels_start + 1))

    for ex in range(m):
        y_i[ex, :] = (np.arange(labels_start, labels_end + 1) == y[ex])

    for layer in layer_sizes:
        layer_size = layer[0] * layer[1]
        layers.append(theta[:layer_size].reshape(layer))
        theta = theta[layer_size:]

    for i in range(len(layer_sizes)):
        a.append(np.hstack((np.ones((m, 1)), X)))
        z.append(a[i] @ layers[i].T)
        X = sigmoid(z[i])

    pos = y_i * np.log(X)
    neg = (1 - y_i) * np.log(1 - X)
    log_error = -pos - neg
    J = np.sum(np.sum(log_error)) / m

    for layer in layers:
        J += np.sum((layer[:, 1:] ** 2)) * (reg / (2 * m))

    deltas = [None for _ in layers]
    deltas[-1] = X - y_i
    for i in range(len(layers) - 2, -1, -1):
        deltas[i] = (deltas[i + 1] @ layers[i + 1])[:, 1:] * sigmoid_gradient(z[i])

    for i in range(len(layers)):
        grads.append(deltas[i].T @ a[i] / m)
        grads[i][:, 1:] += reg * layers[i][:, 1:] / m

    grad = np.hstack([gradient.flatten() for gradient in grads])

    return J, grad


with open('optdigits.csv') as f:
    data = pandas.read_csv(f, header=None)
    data = data.to_numpy()
    np.random.shuffle(data)
    train = data[:-200]
    test = data[-200:]

X = train[:, :-1]
y = train[:, -1]

theta1 = np.random.randn(25, 65) * np.sqrt(2 / 24)
theta2 = np.random.randn(10, 26) * np.sqrt(2 / 9)

layer_sizes = np.array([theta1.shape, theta2.shape])
theta = np.hstack((theta1.flatten(), theta2.flatten()))

theta = opt.fmin_tnc(func=cost, x0=theta, args=(layer_sizes, 9, X, y, 3))[0]

print(np.sum(predict(theta, layer_sizes, test[:, :-1]) == test[:, -1]) / len(test))

for i in range(4):
    for j in range(4):
        if j == 0:
            digit = test[i * 4 + j, :-1].reshape(8, 8)
        else:
            digit = np.hstack((digit, test[i * 4 + j, :-1].reshape(8, 8)))
        x = test[i * 4 + j, :-1].reshape(1, 64)
        plt.text(j * 8, 1 + i * 8, predict(theta, layer_sizes, x)[0], color='white', fontsize=16)
    if i == 0:
        digits = digit
    else:
        digits = np.vstack((digits, digit))

plt.imshow(digits, cmap='gray')
frame1 = plt.gca()
frame1.axes.get_xaxis().set_ticks([-0.5, 7.5, 15.5, 23.5, 31.5])
frame1.axes.get_yaxis().set_ticks([-0.5, 7.5, 15.5, 23.5, 31.5])
for tic in frame1.xaxis.get_major_ticks():
    tic.tick1line.set_visible(False)
    tic.label1.set_visible(False)
for tic in frame1.yaxis.get_major_ticks():
    tic.tick1line.set_visible(False)
    tic.label1.set_visible(False)
plt.grid(color='r', linestyle='-', linewidth=2)
plt.show()
