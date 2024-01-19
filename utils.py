import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler

# print(f"{os.getcwd()}\data")
my_path = f"{os.getcwd()}\data\\train.csv"


###
# plot a scatter plot of coordinates with labels labels
# the data contain k classes
###
def scatter_plot(coordinates, labels, k,title):
    fig, ax = plt.subplots()
    for i in range(k):
        idx = labels == i
        data = coordinates[:, idx]
        ax.scatter(data[0], data[1], label=str(i), alpha=0.3, s=10)
    plt.title(title)
    ax.legend(markerscale=2)
    plt.show()


### FOR THE AUTOENCODER PART
def train(num_epochs, dataloader, model, criterion, optimizer):
    for epoch in range(num_epochs):
        for data in dataloader:
            # print(data)
            img, _ = data
            # print(img)
            # print(f"img len {len(img)}")
            # ===================forward=====================
            output, _ = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))


def get_embedding(model, dataloader):
    model.eval()
    labels = np.zeros((0,))
    embeddings = np.zeros((2, 0))
    for data in dataloader:
        X, Y = data
        with torch.no_grad():
            _, code = model(X)
        embeddings = np.hstack([embeddings, code.numpy().T])
        labels = np.hstack([labels, np.squeeze(Y.numpy())])
    return embeddings, labels


def load_data():
    train = pd.read_csv(my_path)
    labels = train['label']
    data = train.drop(['label'], axis=1)
    return data, labels


def train_data(is_normalize):
    currDir = f'{os.getcwd()}\data'
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))]) if is_normalize else transforms.Compose(
        [transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(root=f'{currDir}', train=True, download=False, transform=transform)
    indices = np.random.permutation(len(train_data))
    train_subset = Subset(train_data, indices=indices[0:10000])
    return train_subset


def autoencoder_data():
    train = train_data(True)
    return torch.utils.data.DataLoader(train, batch_size=32, shuffle=False)
