# CS 181, Spring 2020
# Homework 5: Principal Component Analysis

import numpy as np
import matplotlib.pyplot as plt

# This line loads the images for you. Don't change it!
pics = np.load("data/images.npy")

# Your code here. You may change anything below this line.
class PCA(object):
    # d is the number of principal components
    def __init__(self, d):
        self.d = d
        self.mean = None
        self.standard = None
        self.U = None 
        self.eig = None
        self.Sigma = None
        self.original = None
        

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images. This method should apply PCA to a dataset X.
    def apply(self, X):
        self.original = np.matrix([i.flatten() for i in X])
        self.mean = np.mean(self.original, axis = 0)
        self.standard = np.subtract(self.original, self.mean).T
        self.U, s, vh = np.linalg.svd(self.standard)
        self.Sigma = np.diag(s)
        self.eig = np.square(s)

    def plot_eig(self):
        plt.figure()
        plt.plot(np.arange(self.d), self.eig[0:self.d])
        plt.suptitle("First 500 Eigenvalues by Significance")
        plt.xlabel("PC")
        plt.ylabel("Eigenvalue")
        plt.show()

    def plot_variance(self):
        proportion = np.cumsum(np.divide(self.eig, np.sum(self.eig)))
        plt.figure()
        plt.plot(np.arange(self.d), proportion[0:self.d])
        plt.suptitle("Cumulative Proportion of Variance Explained by First 500 Eigenvectors")
        plt.xlabel("PC")
        plt.ylabel("Cumulative Proportion Variance Explained")
        plt.show()
        print("Variance explained by first 500 components: %f" % np.sum(self.eig[0:500]))

    def plot_mean(self):
        plt.figure()
        plt.imshow(self.mean.reshape(28,28), cmap='Greys_r')
        plt.show()

        fig, ax = plt.subplots(10, 1)
        for i in range(10):
            ax[i].imshow(self.U[:, i].reshape(28,28), cmap='Greys_r')
        plt.show()

    def rec_err(self):
        proj = np.dot(self.standard.T, self.U[:,0:10])
        reconstruct = np.add(self.mean, np.dot(proj, self.U[:,0:10].T))
        error = 0
        for i in range(reconstruct.shape[0]):
            error += np.linalg.norm(np.subtract(self.original[i],reconstruct[i]))
        print("Reconstruction Error Using Mean Image: %f" % error)

        error = 0
        for i in range(reconstruct.shape[0]):
            error += np.linalg.norm(np.subtract(self.standard.T[i],reconstruct[i]))
        print("Reconstruction Error Using first 10 PC: %f" % error)

PCA = PCA(d = 500)
PCA.apply(pics)
PCA.plot_eig()
PCA.plot_variance()
PCA.plot_mean()
PCA.rec_err()
