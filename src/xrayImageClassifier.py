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

    def train(self, n_epochs, batch_size, x_t, y_t, x_v, y_v):

        n_batches = len(x_t) // batch_size

        losses = []
        accuracies = []

        for epoch in range(n_epochs)
            correct = 0.0
            total = 0.0
            epoch_pred = torch.FloatTensor()
            epoch_labels = torch.FloatTensor():
            x_train, y_train = shuffle(x_t, y_t)

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

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
        plt.savefig('output/imageModelTrainingAccuracyLoss.png')
       
        with torch.no_grad():
            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred.round())
            disp.plot()
            plt.savefig('output/imageTestingConfusionMatrix.png')
            
