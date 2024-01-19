# Import Libraries
import utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import numpy as np
import pandas as pd


# https://medium.com/analytics-vidhya/principal-component-analysis-pca-with-code-on-mnist-dataset-da7de0d07c22#:~:text=Save-,Principal%20Component%20Analysis(PCA)%20with%20code%20on%20MNIST%20dataset,the%20threshold%20at%20d%3E3.
# https://www.analyticsvidhya.com/blog/2021/11/pca-on-mnist-dataset/
# https://www.youtube.com/watch?v=OMDn66kM9Qc&ab_channel=LightningAI
# https://www.codingninjas.com/codestudio/library/applying-pca-on-mnist-dataset


def loadmnist():
    global train_loader
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_data, [10000, 50000])
    train_loader = DataLoader(train, batch_size=32)


def PCA(X, components):
    X = StandardScaler().fit_transform(X)  # Normalizing X
    correlation_matrix = X @ X.T  # Correlation matrix
    w, v = eigh(correlation_matrix)  # Eigenvectors
    k = len(v[0]) - components
    v = np.delete(v, list(range(k)), axis=1)
    X_projection = np.matmul(v.T, X)  # Projection matrix
    return X_projection


def main():
    data, labels = utils.load_data()
    data_np = data.to_numpy(dtype="float32").T
    labels_np = labels.to_numpy()
    data_reduced = PCA(data_np, 2)
    utils.scatter_plot(data_reduced, labels_np, 10)


if __name__ == "__main__":
    main()
