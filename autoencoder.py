# Import Libraries
import torch
from torch import nn
import utils as utils
import torch.nn.functional as F



class autoencoder(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(autoencoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_layers[0])
        self.fc2 = torch.nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = torch.nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = torch.nn.Linear(hidden_layers[2], hidden_layers[3])
        self.fc5 = torch.nn.Linear(hidden_layers[3], hidden_layers[2])
        self.fc6 = torch.nn.Linear(hidden_layers[2], hidden_layers[1])
        self.fc7 = torch.nn.Linear(hidden_layers[1], hidden_layers[0])
        self.fc8 = torch.nn.Linear(hidden_layers[0], input_size)

    def forward(self, x):
        x = x.squeeze().view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        y = F.relu(self.fc5(x))
        y = F.relu(self.fc6(y))
        y = F.relu(self.fc7(y))
        y = self.fc8(y).view(-1, 28, 28)
        return y, x



class linear_autoencoder(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(linear_autoencoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_layers[0])
        self.fc2 = torch.nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = torch.nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = torch.nn.Linear(hidden_layers[2], hidden_layers[3])
        self.fc5 = torch.nn.Linear(hidden_layers[3], hidden_layers[2])
        self.fc6 = torch.nn.Linear(hidden_layers[2], hidden_layers[1])
        self.fc7 = torch.nn.Linear(hidden_layers[1], hidden_layers[0])
        self.fc8 = torch.nn.Linear(hidden_layers[0], input_size)

    def forward(self, x):
        x = x.squeeze().view(-1, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        y = self.fc5(x)
        y = self.fc6(y)
        y = self.fc7(y)
        y = self.fc8(y).view(-1, 28, 28)
        return y, x


def run_model(model, figure_title):
    dataloader = utils.autoencoder_data()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    utils.train(30, dataloader, model, criterion, optimizer)
    embeddings, labels = utils.get_embedding(model, dataloader)
    utils.scatter_plot(embeddings, labels, 10, figure_title)

def main():
    autoencoder_model = autoencoder(784, [256, 128, 32, 2])
    linear_autoencoder_model = linear_autoencoder(784, [256, 128, 32, 2])
    run_model(autoencoder_model, " Autoencoder")
    run_model(linear_autoencoder_model, "Linear Autoencoder")


if __name__ == "__main__":
    main()
