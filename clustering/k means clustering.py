import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import gzip

with gzip.open("/content/drive/MyDrive/data/train-labels-idx1-ubyte (1).gz", 'rb') as lbpath:
  y_train = np.frombuffer(lbpath.read(), np.uint8, offset = 8)

with gzip.open('/content/drive/MyDrive/data/train-images-idx3-ubyte (1).gz', 'rb') as imgpath:
  x_train = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(len(y_train), 28, 28)

with gzip.open('/content/drive/MyDrive/data/t10k-labels-idx1-ubyte (1).gz', 'rb') as lbpath:
  y_test = np.frombuffer(lbpath.read(), np.uint8, offset = 8)

with gzip.open('/content/drive/MyDrive/data/t10k-images-idx3-ubyte (1).gz', 'rb') as imgpath:
  x_test = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(len(y_test), 28, 28)


labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train.reshape(len(y_train), 28 * 28)
x_test = x_test.reshape(len(y_test), 28 * 28)

x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

pca = PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)
print(x_train.shape)
print(x_test.shape)

#K means
model = KMeans(n_clusters = 10)
model.fit(x_train)

#K means clustering
def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1-x2)**2))

class kmeans:

  def __init__(self, k, max_iters):
    self.k = k
    self.max_iters = max_iters

    #list of sample indices for each cluster
    self.clusters = [[] for _ in range(self.k)]
    #mean feature vector for each cluster
    self.centroids = []

  def predict(self, x):
    self.x = x
    self.n_samples, self.n_features = x.shape

    #initialize centroids
    random_sample_idxs = np.random.choice(self.n_samples, self.k, replace = False)
    self.centroids = [self.x[idx] for idx in random_sample_idxs]

    #optimization
    for _ in range(self.max_iters):
      #update clusters
      self.clusters = self._create_clusters(self.centroids)
      #update centroids
      centroids_old = self.centroids
      self.centroids = self._get_centroids(self.clusters)
      #check if converged
      if self._is_converged(centroids_old, self.centroids):
        break
    #return cluster labels
    return self._get_cluster_labels(self.clusters)

  def _get_cluster_labels(self, clusters):
    labels = np.empty(self.n_samples)
    for cluster_idx, cluster in enumerate(clusters):
      for sample_idx in cluster:
        labels[sample_idx] = cluster_idx
    return labels

  def _create_clusters(self, centroids):
    clusters = [[] for _ in range(self.k)]
    for idx, sample in enumerate(self.x):
      centroid_idx = self._closest_centroid(sample, centroids)
      clusters[centroid_idx].append(idx)
    return clusters

  def _closest_centroid(self, sample, centroids):
    distances = [euclidean_distance(sample, point) for point in centroids]
    closest_idx = np.argmin(distances)
    return closest_idx

  def _get_centroids(self, clusters):
    centroids = np.zeros((self.k, self.n_features))
    for cluster_idx, cluster in enumerate(clusters):
      cluster_mean = np.mean(self.x[cluster], axis=0)
      centroids[cluster_idx] = cluster_mean
    return centroids

  def _is_converged(self, centroids_old, centroids):
    distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
    return sum(distances) == 0

k = kmeans(10, 100)
k.predict(x_train)
k.predict(x_test)

print(y_train)
print(y_test)
