import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

class oaClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_size = 5

        self.input_layer = nn.Linear(7, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer3 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer4 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.1)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = self.input_layer(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        #x = self.hidden_layer(x)
        #x = F.relu(x)
        #x = self.hidden_layer2(x)
        #x = F.relu(x)
        # x = self.hidden_layer3(x)
        #x = F.relu(x)
        #x = self.hidden_layer4(x)
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
            epoch_loss = 0.0
            correct = 0.0
            total = 0.0
            epoch_pred = torch.FloatTensor()
            epoch_labels = torch.FloatTensor()


            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                x = Variable(torch.FloatTensor(x_train.values[start:end]))
                x = torch.nan_to_num(x, nan=0.0)
                y = Variable(torch.FloatTensor(y_train.values[start:end]))
                y = torch.nan_to_num(y, nan=0.0)
                y = y.unsqueeze(1)

                self.optimizer.zero_grad()
                
                y_pred = self(x)

                loss = self.criterion(y_pred, y)

                loss.backward()
                self.optimizer.step()

                epoch_pred = torch.cat((epoch_pred, y_pred))
                epoch_labels = torch.cat((epoch_labels, y))

                epoch_loss += loss.item()
                correct += (y_pred.round() == y).sum().item()
                total += len(y)

            self.scheduler.step(epoch_loss)

            accuracy = correct / total
            losses.append(epoch_loss)
            accuracies.append(accuracy)
            print(f"Epoch {epoch} loss = {epoch_loss}; accuracy = {accuracy:.2%} | {correct} / {total}")

            if epoch == (n_epochs - 1):
                with torch.no_grad():
                    disp = ConfusionMatrixDisplay.from_predictions(epoch_labels, epoch_pred.round())
                    disp.plot()
                    plt.savefig('output/trainingConfusionMatrix.png')

        torch.no_grad()

        x_test, y_test = shuffle(x_v, y_v)
        x_test = Variable(torch.FloatTensor(x_test.values[:]))
        x_test = torch.nan_to_num(x_test, nan=0.0)
        y_test = torch.FloatTensor(y_test.to_numpy())
        y_test = torch.unsqueeze(y_test, 1)
        y_test = torch.nan_to_num(y_test, nan=0.0)

        y_pred = self(x_test)
        correct = (y_pred.round() == y_test).sum().item()
        accuracy = correct / len(y_test)

        print(f"Test accuracy = {accuracy} | Total = {len(y_test)} correct")

        plt.figure()
        plt.title("Training Results")
        plt.plot(losses, label="Losses", linestyle='--')
        plt.plot(accuracies, label="Accuracy", linestyle='-')
        ticks = list(range(0, n_epochs, 5))
        ticks.append(n_epochs - 1)
        plt.xticks(ticks)
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig('output/modelTrainingAccuracyLoss.png')
       
        with torch.no_grad():
            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred.round())
            disp.plot()
            plt.savefig('output/testingConfusionMatrix.png')
