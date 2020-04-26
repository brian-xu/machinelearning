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


def cost(theta: np.array, layer_sizes: np.array, labels: int, X: np.array, y: np.array, reg: float = 0) -> (float, np.array):
    """
    theta: unrolled array containing all layers
    layer_sizes: array indicating the size of layers
    labels: integer indicating the number of labels in y
    X: m x n input layer
    y: m x 1 output with values ranging from 0-labels
    reg: regularization parameter
    """
    m, n = X.shape
    layers = []
    a = []
    z = []
    grads = []
    y_i = np.zeros((m, labels + 1))

    for ex in range(m):
        y_i[ex, :] = (np.arange(labels+1) == y[ex])

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
    J = np.sum(np.sum(-pos - neg)) / m

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


with open('optdigits.txt') as f:
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
