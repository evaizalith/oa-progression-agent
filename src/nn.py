import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

class oaClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # We make use of input data with 47 features (53 after transformation)
        # Our output is a the probability of progressor or non-progressor
        self.input_layer = nn.Linear(6, 5)
        self.hidden_layer = nn.Linear(5, 5)
        self.hidden_layer2 = nn.Linear(5, 5)
        self.hidden_layer3 = nn.Linear(5, 5)
        self.hidden_layer4 = nn.Linear(5, 5)
        self.output_layer = nn.Linear(5, 1)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

    def forward(self, x):
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]

        x = self.input_layer(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.hidden_layer2(x)
        x = F.relu(x)
        x = self.hidden_layer3(x)
        x = F.relu(x)
        x = self.hidden_layer4(x)
        x = F.relu(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x

    def train(self, n_epochs, batch_size, x_t, y_t, x_v, y_v):
        
        n_batches = len(x_t) // batch_size

        losses = []
        accuracies = []

        for epoch in range(n_epochs):
            x_train, y_train = shuffle(x_t, y_t)
            y_train = torch.FloatTensor(y_train.to_numpy())
            y_train = torch.unsqueeze(y_train, 1)
            epoch_loss = 0.0
            correct = 0.0
            total = 0.0

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                x = Variable(torch.FloatTensor(x_train.values[start:end]))
                x = torch.nan_to_num(x, nan=0.0)
                y = y_train[start:end, :]
                y = torch.nan_to_num(y, nan=0.0)

                self.optimizer.zero_grad()
                
                y_pred = self(x)

                loss = self.criterion(y_pred, y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                correct += (y_pred.round() == y).sum().item()
                total += len(y)

            self.scheduler.step()

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

        y_pred = self(x_test)
        #correct = (y_pred.round() == y_test[0]).sum().item()
        #total = len(y_test)

        _, y_pred_tags = torch.max(y_pred, dim = 1)  

        _, y_test_tag= torch.max(y_test, dim = 1)

        correct_pred = (y_pred_tags == y_test_tag).float()

        acc = correct_pred.sum() / len(correct_pred)

        accuracy = torch.round(acc * 100)

        #accuracy = correct / total
        #print(f"Testing accuracy: {accuracy:.2%} | {correct} / {total}")
        print(f"Test accuracy = {accuracy}")

        plt.figure()
        plt.title("Training Results")
        plt.plot(losses, label="Losses", linestyle='--')
        plt.plot(accuracies, label="Accuracy", linestyle='-')
        #plt.yticks(np.arange(0, max(losses)) + 1, 0.1)
        plt.xticks(range(0, n_epochs))
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
