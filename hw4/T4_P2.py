# CS 181, Spring 2020
# Homework 4

import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns; sns.set()

# This line loads the images for you. Don't change it! 
large_dataset = np.load("data/large_dataset.npy").astype(np.int64)
small_dataset = np.load("data/small_dataset.npy").astype(np.int64)
small_labels = np.load("data/small_dataset_labels.npy").astype(int)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# Keep in mind you may add more public methods for things like the visualization.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.

class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.means = [np.random.rand(28, 28) for i in range(K)]
        self.costs = []
        self.cls = []

    def L2norm(self, u, v):
        return np.linalg.norm(np.subtract(u,v))

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
    def fit(self, X):
        classes = [0 for i in range(X.shape[0])]
		
        while True:
            changes = 0
            for i in range(X.shape[0]):
                xc = X[i].reshape(28, 28)
                distances = [self.L2norm(xc, mean) for mean in self.means]
                update = np.argmin(distances)
                if classes[i] != update:
                    classes[i] = update
                    changes += 1

            for i in range(self.K):
                if len(X[np.array(classes) == i]) != 0:
                    self.means[i] = np.mean(X[np.array(classes) == i], axis=0).reshape(28,28)
                    
            cost = 0
            for i in range(X.shape[0]):
                xc = X[i].reshape(28, 28)
                cost += (self.L2norm(self.means[classes[i]], xc))**2
            self.costs.append(cost)
			
            if changes == 0:
                break
        self.cls = classes
        

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.means

    def visualize(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.suptitle("Objective Function vs. Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Objective Function")
        plt.show()

    def avg_obj(self):
        return self.costs[-1]

# Part 1
K = 10
KMeansClassifier = KMeans(K=10)
KMeansClassifier.fit(large_dataset)
KMeansClassifier.visualize()

# Part 2
def eb(K, restart):
    costs = []
    sd = []
    for i in K:
        costs_k = []
        for j in range(restart):
            KMeansClassifier = KMeans(K=i)
            KMeansClassifier.fit(large_dataset)
            costs_k.append(KMeansClassifier.avg_obj())
        costs.append(np.mean(costs_k))
        sd.append(np.std(costs_k))

    plt.figure()
    plt.errorbar(K, costs, yerr = sd, fmt='o')
    plt.suptitle("Average Objective vs. K")
    plt.xlabel("K")
    plt.ylabel("Average Objective Function")
    plt.show()

eb([5,10,20], 5)

# Part 3
fig, ax = plt.subplots(10, 5)
for i in range(5):
    KMeansClassifier = KMeans(K=K)
    KMeansClassifier.fit(large_dataset)

    images = np.array(KMeansClassifier.get_mean_images()).reshape(K, 784)
    for k in range(10):
        ax[k, i].imshow(images[k].reshape(28,28), cmap='Greys_r')
plt.show()

# Part 4

standard = [None for i in range(large_dataset.shape[0])]

mean = np.mean(large_dataset, axis = 0)
sd = np.std(large_dataset, axis = 0)
sd[sd == 0] = 1

for i, X in enumerate(large_dataset):
    standard[i] = np.divide(np.subtract(X, mean), sd)
standard = np.array(standard) 


fig, ax = plt.subplots(10, 5)
for i in range(5):
    KMeansClassifier = KMeans(K=K)
    KMeansClassifier.fit(standard)
    images = np.array(KMeansClassifier.get_mean_images()).reshape(K, 784)

    for k in range(10):
        ax[k, i].imshow(images[k].reshape(28,28), cmap='Greys_r')
plt.show()

# Part 5
class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage
        self.means = [None for i in range(10)]
        self.classes = []
        self.merges = []
        self.mindist = []
        self.cls = []

    def dist(self, A):
        temp = np.empty([len(A), len(A)])
        if self.linkage == 'min':
            for i in range(len(A)):
                for j in range(len(A)):
                    if A[i].ndim != 1 or A[j].ndim != 1:
                        stemp = distance.cdist(np.atleast_2d(A[i]), np.atleast_2d(A[j]), 'euclidean')
                        masked = np.ma.MaskedArray(stemp, stemp <= 0)
                        idx = np.unravel_index(np.ma.argmin(masked), (masked.shape[0], masked.shape[1]))
                        temp[i][j] = masked[idx[0]][idx[1]]
                    else:
                        temp[i][j] = distance.cdist(np.atleast_2d(A[i]), np.atleast_2d(A[j]), 'euclidean')
        elif self.linkage == 'max':
            for i in range(len(A)):
                for j in range(len(A)):
                    if A[i].ndim != 1 or A[j].ndim != 1:
                        stemp = distance.cdist(np.atleast_2d(A[i]), np.atleast_2d(A[j]), 'euclidean')
                        masked = np.ma.MaskedArray(stemp, stemp <= 0)
                        idx= np.unravel_index(np.ma.argmax(masked), (masked.shape[0], masked.shape[1]))
                        temp[i][j] = masked[idx[0]][idx[1]]
                    else:
                        temp[i][j] = distance.cdist(np.atleast_2d(A[i]), np.atleast_2d(A[j]), 'euclidean')
        elif self.linkage == 'centroid':
            for i in range(len(A)):
                for j in range(len(A)):
                    if A[i].ndim != 1:
                        itemp = np.mean(A[i], axis = 0)
                    else:
                        itemp = A[i]
                    if A[j].ndim != 1:
                        jtemp = np.mean(A[j], axis = 0)
                    else:
                        jtemp = A[j]
                    temp[i][j] = distance.cdist(np.atleast_2d(itemp), np.atleast_2d(jtemp), 'euclidean')
        return temp

    def fit(self, X):
        j = 1
        self.cls = [0 for i in range(X.shape[0])]
        self.classes = [X[i] for i in range(X.shape[0])]
        while len(self.classes) != 10:
            distances = self.dist(self.classes)
            np.fill_diagonal(distances, np.inf)
            idx = np.unravel_index(np.argmin(distances), (distances.shape[0], distances.shape[1]))
            temp = np.vstack((self.classes[idx[0]], self.classes[idx[1]]))
            self.merges.append(j)
            self.mindist.append(distances[idx[0]][idx[1]])
            self.classes[idx[0]] = temp
            self.cls[idx[0]] = idx[0]
            self.cls[idx[1]]= idx[1]
            self.classes.pop(idx[1])
            j += 1

        for i in range(len(self.classes)):
            if self.classes[i].ndim != 1:
                self.means[i] = np.mean(self.classes[i], axis=0).reshape(28, 28)
            else:
                self.means[i] = self.classes[i].reshape(28,28)

        i = 0
        for t in range(len(self.classes)):
            if(len(self.classes[t].shape) == 1):
                length = 1
            else:
                length = self.classes[t].shape[0]
            for g in range(length):
                self.cls[i] = t
                i += 1

    def get_mean_images(self):
        return self.means

    def visualize(self):
        plt.figure()
        plt.plot(np.arange(self.merges[0], self.merges[-1] + 1, 1), self.mindist)
        plt.suptitle("# of Merges vs. Distance (%s)" % self.linkage)
        plt.xlabel("Total Number of Merges Completed")
        plt.ylabel("Distance Between Most Recently Merged Clusters")

# Part 5, 6
fig, ax = plt.subplots(10, 3)
j = 0
for i in ['min', 'max', 'centroid']:
    HA = HAC(linkage = i)
    HA.fit(small_dataset)
    images = HA.get_mean_images()
    for k in range(10):
        ax[k, j].imshow(images[k], cmap='Greys_r')
    HA.visualize()
    j += 1
plt.show()

# Part 7
KMeansClassifier = KMeans(K=K)
KMeansClassifier.fit(small_dataset)
sclass = np.array(KMeansClassifier.cls)

conf = np.zeros((K, K))
for j in range(K):
    for i in range(len(sclass)):
        if(small_labels[i] == j):
            if (sclass[i] != j):
                conf[sclass[i]][j] += 1
sns.heatmap(conf)
plt.show()

print(small_dataset[1:20])
for i in ['min', 'max', 'centroid']:
    HA = HAC(linkage = i)
    HA.fit(small_dataset)
    sclass = np.array(HA.cls)
    conf = np.zeros((K, K))
    for j in range(K):
        for i in range(len(sclass)):
            if(small_labels[i] == j):
                if (sclass[i] != j):
                    conf[sclass[i]][j] += 1
    sns.heatmap(conf)
    plt.show()
    
