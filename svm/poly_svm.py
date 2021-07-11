import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import gzip

#load data 
with gzip.open('C:/Users/USER/Documents/git/my/data/train-labels-idx1-ubyte (1).gz', 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset = 8)


with gzip.open('C:/Users/USER/Documents/git/my/data/train-images-idx3-ubyte (1).gz', 'rb') as imgpath:
    x_train = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(len(y_train), 28, 28)

with gzip.open('C:/Users/USER/Documents/git/my/data/t10k-labels-idx1-ubyte (1).gz', 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset = 8)

with gzip.open('C:/Users/USER/Documents/git/my/data/t10k-images-idx3-ubyte (1).gz', 'rb') as imgpath:
    x_test = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(len(y_test), 28, 28)


labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#3D array to 2D array 
x_train = x_train.reshape(len(y_train), 28 * 28)
x_test = x_test.reshape(len(y_test), 28 * 28)

#create model & train
PolySVM = SVC(kernel = 'poly', C = 1.0, gamma = 0.1, degree = 6, probability = True, random_state = None, max_iter = 1000)
y_pred = PolySVM.predict(x_test)

accuracy_score = PolySVM.score(x_test, y_test)
print(accuracy_score)

x_test = x_test.reshape(len(y_test), 28, 28)

for i in range(10):
    plt.imshow(x_test[i], cmap = plt.cm.get_cmap("binary"))
    print("Predict: ", y_pred[i])
    plt.imshow()
