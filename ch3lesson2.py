import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

X, y = spiral_data(100, 3)


plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.show()

