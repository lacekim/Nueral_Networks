import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes, 2)) # list of given number of points per each class, containing pairs of values
    y = np.zeros(points*classes, dtype='uint8')  # same as above, but containing simple values - classes
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))  # index in class
        X[ix] = np.c_[np.random.randn(points)*.1 + class_number/3, np.random.randn(points)*.1 + 0.5]
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