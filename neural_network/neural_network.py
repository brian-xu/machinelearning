import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(x: np.array) -> np.array:
    """
    Returns the sigmoid transformation of each value in an array.
    x: m x n array
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x: np.array) -> np.array:
    """
    Returns the partial derivative of the sigmoid for each value in an array.
    x: m x n array
    """
    return sigmoid(x) * (1 - sigmoid(x))


def reshape_theta(theta: np.array, layer_sizes: np.array) -> [np.array, ]:
    """
    Reforms individual layers as defined by layer sizes.
    theta: unrolled array containing all layers
    layer_sizes: array indicating the size of layers
    """
    layers = []

    for layer in layer_sizes:
        layer_size = layer[0] * layer[1]
        layers.append(theta[:layer_size].reshape(layer))
        theta = theta[layer_size:]

    return layers


def forward_propagate(layers: [np.array, ], x: np.array) -> [[np.array, ], ]:
    """
    Perform forwards propagation with the given thetas and input layer.
    layers: array containing thetas as individual arrays.
    x: m x n array
    """
    a = []
    z = []
    m, n = x.shape

    for i in range(len(layers)):
        a.append(np.hstack((np.ones((m, 1)), x)))
        z.append(a[i] @ layers[i].T)
        x = sigmoid(z[i])

    return a, z, x


def back_propagate(a: [np.array, ], z: [np.array, ], x: np.array,
                   y_i: np.array, layers: [np.array], reg: float) -> np.array:
    """
    Perform backwards propagation to calculate layer gradients with the given thetas and output.
    layers: array containing thetas as individual arrays.
    x: m x n array
    """
    m, n = x.shape

    grads = []
    deltas = [None for _ in layers]
    deltas[-1] = x - y_i
    for i in range(len(layers) - 2, -1, -1):
        deltas[i] = (deltas[i + 1] @ layers[i + 1])[:, 1:] * sigmoid_gradient(z[i])

    for i in range(len(layers)):
        grads.append(deltas[i].T @ a[i] / m)
        grads[i][:, 1:] += reg * layers[i][:, 1:] / m

    grad = np.hstack([gradient.flatten() for gradient in grads])
    return grad


def predict(theta: np.array, layer_sizes: np.array, x: np.array) -> np.array:
    """
    Return the index of the maximum value at every row.
    theta: unrolled array containing all layers
    layer_sizes: array indicating the size of layers
    x: m x n array
    """
    layers = reshape_theta(theta, layer_sizes)
    _, _, x = forward_propagate(layers, x)
    return np.argmax(x, axis=1)


def cost(theta: np.array, layer_sizes: np.array, labels_end: int,
         x: np.array, y: np.array, reg: float = 0, labels_start: int = 0) -> (float, np.array):
    """
    theta: unrolled array containing all layers
    layer_sizes: array indicating the size of layers
    labels_end: integer indicating the largest label in y
    x: m x n input layer
    y: m x 1 output with values ranging from 0-labels
    reg: regularization parameter
    labels_start: optional argument indicating the smallest label in y
    """
    m, n = x.shape
    layers = reshape_theta(theta, layer_sizes)
    y_i = np.zeros((m, labels_end - labels_start + 1))

    for ex in range(m):
        y_i[ex, :] = (np.arange(labels_start, labels_end + 1) == y[ex])

    a, z, x = forward_propagate(layers, x)

    pos = y_i * np.log(x)
    neg = (1 - y_i) * np.log(1 - x)
    log_error = -pos - neg
    J = np.sum(np.sum(log_error)) / m

    for layer in layers:
        J += np.sum((layer[:, 1:] ** 2)) * (reg / (2 * m))

    grad = back_propagate(a, z, x, y_i, layers, reg)
    return J, grad


def initialize_weights(shape: (int, int)) -> np.array:
    """
    Initialize weight layers with a random non-zero value to prevent convergence from stalling.
    shape: 1 x 2 array
    """
    m, n = shape
    return np.random.randn(m, n) * np.sqrt(2 / (m - 1))


with open('optdigits.csv') as f:
    data = pd.read_csv(f, header=None)
    data = data.to_numpy()
    np.random.shuffle(data)
    test_len = int(np.round(len(data) * 3 / 10))
    train = data[:-test_len]
    test = data[-test_len:]

x = train[:, :-1]
y = train[:, -1]

theta1 = initialize_weights((25, 65))
theta2 = initialize_weights((10, 26))

layer_sizes = np.array([theta1.shape, theta2.shape])
theta = np.hstack((theta1.flatten(), theta2.flatten()))

theta = opt.fmin_tnc(func=cost, x0=theta, args=(layer_sizes, 9, x, y, 3))[0]

p_x = predict(theta, layer_sizes, test[:, :-1])

print("Test set accuracy:", np.sum(p_x == test[:, -1]) / len(test))

# Below code is not vectorized but it's just visualization code

fig, ax = plt.subplots(4, 4)

for i in range(4):
    for j in range(4):
        digit = test[i * 4 + j, :-1].reshape(8, 8)
        digit_x = digit.reshape(1, 64)
        ax[i, j].imshow(digit, cmap='gray')
        ax[i, j].set_title(f'Predicted: {predict(theta, layer_sizes, digit_x)[0]} ({test[i * 4 + j, -1]})')
        ax[i, j].axis('off')

fig.show()
