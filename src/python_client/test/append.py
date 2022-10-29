import numpy as np

empty_array = np.empty(shape=(0, 4))
print(empty_array)
empty_array = np.append(empty_array, np.array([[1, 2, 3, 4]]), axis=0)
print(empty_array)
