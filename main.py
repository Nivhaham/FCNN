import random

from torchvision import datasets as dts
from torchvision.transforms import ToTensor
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

def main():
    train_data = datasets.MNIST(root='./dat', train=True, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_data, [10000, 50000])
    train_loader = DataLoader(train, batch_size=32)
    for data in train_loader: # run over pictures
        print(data[0].shape)
        break

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
