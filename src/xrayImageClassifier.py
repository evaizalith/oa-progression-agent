import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

class imageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 32 * 32, 1)
        #self.fc2 = nn.Linear(50, 30)
        #self.fc3 = nn.Linear(50, 1)
        
    def forward(self, x):
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[2, 3])

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        #x = self.fc3(x)
        x = torch.sigmoid(x)

        return x

class xrayImageClassifier():
    def __init__(self):
        self.net = imageNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()

    def train(self, n_epochs, batch_size, x_t, y_t, x_v, y_v):

        #n_batches = len(x_t) // batch_size
        n_batches = 200 // batch_size

        losses = []
        accuracies = []

        for epoch in range(n_epochs):
            correct = 0.0
            total = 0.0
            epoch_loss = 0.0
            epoch_pred = torch.FloatTensor()
            epoch_labels = torch.FloatTensor()
            y_t = Variable(torch.FloatTensor(y_t))
            y_t.unsqueeze(1)
            x_train, y_train = shuffle(x_t, y_t)

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                x = x_train[start:end]
                y = y_train[start:end].unsqueeze(1)

                self.optimizer.zero_grad()
                
                y_pred = self.net(x)

                loss = self.criterion(y_pred, y)

                loss.backward()
                self.optimizer.step()

                epoch_pred = torch.cat((epoch_pred, y_pred))
                epoch_labels = torch.cat((epoch_labels, y))

                epoch_loss += loss.item()
                correct += (y_pred.round() == y).sum().item()
                total += len(y)

            accuracy = correct / total
            losses.append(epoch_loss)
            accuracies.append(accuracy)
            print(f"Epoch {epoch} loss = {epoch_loss}; accuracy = {accuracy:.2%} | {correct} / {total}")

            if epoch == (n_epochs - 1):
                with torch.no_grad():
                    disp = ConfusionMatrixDisplay.from_predictions(epoch_labels, epoch_pred.round())
                    disp.plot()
                    plt.savefig('output/imageTrainingConfusionMatrix.png')

                    plt.clf()

                    auc = roc_auc_score(epoch_labels, epoch_pred)
                    fpr, tpr, thresholds = roc_curve(epoch_labels, epoch_pred)
                    plt.plot(fpr, tpr, label='AUC = %0.2f' % auc)
                    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line representing random chance
                    plt.xlabel('Specificity (False Positive Rate)')
                    plt.ylabel('Sensitivity (True Positive Rate)')
                    plt.title('AUC')
                    plt.legend(loc="lower right")
                    plt.savefig('output/imageAUC.png')

        torch.no_grad()

        x_test, y_test = shuffle(x_v, y_v)
        y_test = Variable(torch.FloatTensor(y_test))

        y_pred = self.net(x_test)
        correct = (y_pred.round() == y_test)
        print(correct)
        print(len(y_v))
        accuracy = correct / len(y_v)

        print(f"Test accuracy = {accuracy} | Total = {len(y_v)}")

        plt.figure()
        plt.title("Training Results")
        plt.plot(losses, label="Losses", linestyle='--')
        plt.plot(accuracies, label="Accuracy", linestyle='-')
        ticks = list(range(0, n_epochs, 5))
        ticks.append(n_epochs - 1)
        plt.xticks(ticks)
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig('output/imageModelTrainingAccuracyLoss.png')
       
        with torch.no_grad():
            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred.round())
            disp.plot()
            plt.savefig('output/imageTestingConfusionMatrix.png')
            
