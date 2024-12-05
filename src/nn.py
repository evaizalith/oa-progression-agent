import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable
from sklearn.utils import shuffle

class oaClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # We make use of input data with 47 features (53 after transformation)
        # Our output is a the probability of progressor or non-progressor
        self.input_layer = nn.Linear(6, 4)
        self.hidden_layer = nn.Linear(4, 4)
        self.output_layer = nn.Linear(4, 1)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x

    def train(self, n_epochs, batch_size, x_t, y_t):
        
        n_batches = len(x_t) // batch_size

        for epoch in range(n_epochs):
            x_train, y_train = shuffle(x_t, y_t)
            y_train = torch.FloatTensor(y_train.to_numpy())
            y_train = torch.unsqueeze(y_train, 1)
            epoch_loss = 0.0
            for i in range(n_batches):

                x0 = i * batch_size
                xN = x0 + batch_size

                x = Variable(torch.FloatTensor(x_train.values[x0:xN]))
                x = torch.nan_to_num(x, nan=0.0)
                y = y_train[x0:xN, :]
                y = torch.nan_to_num(y, nan=0.0)

                self.optimizer.zero_grad()
                
                y_pred = self(x)

                loss = self.criterion(y_pred, y)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch} loss = {epoch_loss}")
