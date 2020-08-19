import numpy as np
import matplotlib.pyplot as plt

from ch4lesson1 import Activation_ReLU, Layer_Dense, Activation_Softmax
from ch5lesson1 import Loss_CategoricalCrossentropy

np.random.seed(0)


def create_data(points, classes):
    X = np.zeros((points * classes, 2))  # list of given number of points per each class, containing pairs of values
    y = np.zeros(points * classes, dtype='uint8')  # same as above, but containing simple values - classes
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))  # index in class
        X[ix] = np.c_[np.random.randn(points) * .1 + class_number / 3, np.random.randn(points) * .1 + 0.5]
        y[ix] = class_number
    return X, y


X, y = create_data(100, 3)

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()


# Create dataset
X, y = create_data(100, 3)

# Create model
dense1 = Layer_Dense(2, 3)  # first dense layer, 2 inputs (each sample has 2 features), 3 outputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)  # second dense layer, 3 inputs, 3 outputs
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 9999999  # some initial value
best_dense1_weights = dense1.weights
best_dense1_biases = dense1.biases
best_dense2_weights = dense2.weights
best_dense2_biases = dense2.biases

for iteration in range(10000):

    # Generate a new set of weights for iteration
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.randn(1, 3)

    # Make a forward pass of the training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss (from activation output, softmax activation here) and accuracy
    loss = loss_function.forward(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
    accuracy = np.mean(predictions==y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration, 'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights
        best_dense1_biases = dense1.biases
        best_dense2_weights = dense2.weights
        best_dense2_biases = dense2.biases
        lowest_loss = loss