import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 

#npy file load
train_data = np.load("regression.npy")
test_data = np.load("regression_test.npy")

#train & test data
X_train = np.array(train_data[:,0])
y_train = np.array(train_data[:,1])

X_train = np.expand_dims(X_train, axis = -1)
y_train = np.expand_dims(y_train, axis = -1)

x_test = np.array(test_data[:,0])
y_test = np.array(test_data[:,1])

x_test = np.expand_dims(x_test, axis = -1)
y_test = np.expand_dims(y_test, axis = -1)

#model
pipeline = make_pipeline(PolynomialFeatures(4), LinearRegression())
pipeline.fit(np.array(X_train), y_train)

y_predicted = pipeline.predict(x_test)

accuracy_score = pipeline.score(X_train, y_train)
print(accuracy_score)

df = pd.DataFrame({'x': x_test[:,0], 'y':y_predicted[:,0]})
df.sort_values(by='x', inplace = True)
points = pd.DataFrame(df).to_numpy()

plt.scatter(x_test, y_test)
plt.plot(points[:,0], points[:,1], color = 'blue', linewidth = 3)
plt.title('Polynomial Regression')
plt.show()
