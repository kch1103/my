import torch 
import torchvision
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat 
from sklearn.preprocessing import LabelBinarizer
from torchvision import models
from torchsummary import summary

resnet18_pretrained = models.resnet18(pretrained = True)
print(resnet18_pretrained)

train_raw = loadmat('/content/drive/MyDrive/data/train_32x32.mat')
test_raw = loadmat('/content/drive/MyDrive/data/test_32x32.mat')

train_images = np.array(train_raw['X'])
test_images = np.array(test_raw['X'])

train_labels = train_raw['y']
test_labels = test_raw['y']

print(train_images.shape)
print(test_images.shape)

print(train_labels)
print(test_labels)

train_images = np.moveaxis(train_images, -1, 0)
test_images = np.moveaxis(test_images, -1, 0)

print(train_images.shape)
print(test_images.shape)

train_images = torch.FloatTensor(train_images)
test_images = torch.FloatTensor(test_images)

lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.fit_transform(test_labels)

train_labels = torch.FloatTensor(train_labels)
test_labels = torch.FloatTensor(test_labels)

train_images /= 255.0
test_images /= 255.0

learning_rate = 0.001
batch_size = 50
epochs = 100

train_images = torch.utils.data.DataLoader(dataset = train_images,
                                           batch_size = batch_size,
                                           shuffle = True
                                           )
test_images = torch.utils.data.DataLoader(dataset = test_images,
                                          batch_size = batch_size,
                                          shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

num_classes = 10
num_ftrs = resnet18_pretrained.fc.in_features
resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)

resnet18_pretrained.to(device)
model = resnet18_pretrained

train_images = (torch.Tensor(73257, 3, 32, 32)).to(device)
test_images = (torch.Tensor(26032, 3, 32, 32)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
  for batch_idx, (data, targets) in enumerate(train_images):
    data = data.to(device=device)
    targets = targets.to(device=device)

    data = data.reshape(data.shape[0], -1)
    
    #forward 
    scores = model(data)
    loss = criterion(scores, targets)

    #backward
    optimizer.zero_grad()
    loss.backward()

    #optimizer
    optimizer.step()

with torch.no_grad():
  prediction = model(test_images)
  accuracy = model.score(test_images, test_labels)
  print('Accuracy :', accuracy)
