import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


#npy file load
train_data = np.load("regression.npy")
test_data = np.load("regression_test.npy")

#train & test data
X_train = np.array(train_data[:,0])
y_train = np.array(train_data[:,1])

X_train = np.expand_dims(X_train, axis = -1) 
#y_train = np.expand_dims(y_train, axis = -1)

x_test = np.array(test_data[:,0])
y_test = np.array(test_data[:,1])

x_test = np.expand_dims(x_test, axis = -1)
#y_test = np.expand_dims(y_test, axis = -1)

#binary
T1 = np.mean(y_train)
T2 = np.mean(y_test)

upper, lower = 1, 0

y_train = np.where(y_train>T1, upper, lower)
y_test = np.where(y_test>T2, upper, lower)

print(y_train)
print(y_test)

#model
model = LogisticRegression()
model.fit(X_train, y_train)

y_predicted = model.predict(x_test)

accuracy_score = model.score(X_train, y_train)
print(accuracy_score)

plt.scatter(x_test, y_test)
plt.plot(x_test, y_predicted, color = 'blue', linewidth = 3)
plt.title('Logistic Regression')
plt.show()
