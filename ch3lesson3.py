import numpy as np
import nnfs

nnfs.init()

inputs = 2
neurons = 4

weights = 0.01 * np.random.randn(inputs, neurons)
biases = np.zeros((1, neurons))

print(weights)
print(biases)


