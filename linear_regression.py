import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#npy file load
train_data = np.load("regression.npy")
test_data = np.load("regression_test.npy")

#train & test data
X_train = np.c_[train_data[:,0]]
y_train = np.c_[train_data[:,1]]

x_test = np.c_[test_data[:,0]]
y_test = np.c_[test_data[:,1]]

#model
model = LinearRegression()
model.fit(X_train, y_train)

y_predicted = model.predict(x_test)

accuracy_score = model.score(X_train, y_train)
print(accuracy_score)

plt.scatter(x_test, y_test)
plt.plot(x_test, y_predicted, color = 'blue', linewidth = 3)
plt.title('Linear Regression')
plt.show()
