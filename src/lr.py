import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 1)

    def forward(self, x):
        x = self.linear(x)
        x = F.sigmoid(x)
        return x

class XrayLR():
    def __init__(self):
        self.lr = LogisticRegression()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.lr.parameters(), lr=0.1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, n_epochs, batch_size, x_t, y_t, x_v, y_v):
        n_batches = len(x_t) // batch_size

        losses = []
        accuracies = []

        for epoch in range(n_epochs):
            x_train, y_train = shuffle(x_t, y_t)
            epoch_loss = 0.0
            correct = 0.0
            total = 0.0

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                x = Variable(torch.FloatTensor(x_train.values[start:end]))
                x = torch.nan_to_num(x, nan=0.0)
                y = Variable(torch.FloatTensor(y_train.values[start:end]))
                y = torch.nan_to_num(y, nan=0.0)
                y = y.unsqueeze(1)

                self.optimizer.zero_grad()
                
                y_pred = self.lr(x)

                loss = self.criterion(y_pred, y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                correct += (y_pred.round() == y).sum().item()
                total += len(y)

            #self.scheduler.step(epoch_loss)

            accuracy = correct / total
            losses.append(epoch_loss)
            accuracies.append(accuracy)
            print(f"Epoch {epoch} loss = {epoch_loss}; accuracy = {accuracy:.2%} | {correct} / {total}")

        torch.no_grad()

        x_test, y_test = shuffle(x_v, y_v)
        x_test = Variable(torch.FloatTensor(x_test.values[:]))
        x_test = torch.nan_to_num(x_test, nan=0.0)
        y_test = torch.FloatTensor(y_test.to_numpy())
        y_test = torch.unsqueeze(y_test, 1)
        y_test = torch.nan_to_num(y_test, nan=0.0)

        y_pred = self.lr(x_test)
        correct = (y_pred.round() == y_test).sum().item()
        accuracy = correct / len(y_test)

        print(f"Test accuracy = {accuracy} | Total = {len(y_test)} correct")
        #print(y_test)
        #print(y_pred)

        plt.figure()
        plt.title("Training Results")
        plt.plot(losses, label="Losses", linestyle='--')
        plt.plot(accuracies, label="Accuracy", linestyle='-')
        plt.xticks(range(0, n_epochs))
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()         
