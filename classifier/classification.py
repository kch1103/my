import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC 
import gzip
import argparse

#argparse
parser = argparse.ArgumentParser()
parser.add_argument('--classifier', type = str)
args = parser.parse_args()

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
classifier = {
    'LinearSVM': LinearSVC(),
    'KNN' : KNeighborsClassifier(n_neighbors = 100),
    'GaussianNB': GaussianNB(),
    'RandomForest': RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 0)
    }

classifier[args.classifier].fit(x_train, y_train)   
y_pred = classifier[args.classifier].predict(x_test)
accuracy_score = classifier[args.classifier].score(x_test, y_test)
print(accuracy_score)

x_test = x_test.reshape(len(y_test), 28, 28)

for i in range(10):
    plt.imshow(x_test[i], cmap = plt.cm.get_cmap("binary"))
    print("Predict: ", y_pred[i])
    plt.imshow()
