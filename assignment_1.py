import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


train_data = np.load("C:/Users/USER/Documents/assignment_1/regression.npy")
test_data = np.load("C:/Users/USER/Documents/assignment_1/regression_test.npy")

X_train = np.c_[train_data[:,0]]
y_train = np.c_[train_data[:,1]]

x_test = test_data[:,0]
y_test = test_data[:,0:1]

plt.scatter(X_train, y_train)
plt.show()


