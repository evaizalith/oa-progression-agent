import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class imageDataset():
    def load(filepath):

    def normalize():

class imageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d
        self.conv2 = nn.Conv2d
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear
        self.fc2 = nn.Linear
        self.fc3 = nn.Linear
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x

class xrayImageClassifier():
    def __init__(self):
        self.net = imageNet()
        self.optim = optim.SGD(self.net.parameters(), lr=0.01)
        self.criterion == nn.CrossEntropyLoss()

    def train(self, data, n_epochs, batch_size):
        accuracies = []
        losses = []

        for epoch in range(n_epochs):
    
