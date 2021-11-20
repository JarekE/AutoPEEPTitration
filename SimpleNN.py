# Simple NN
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch import nn, optim
import torch.nn.functional as F
import config


class Net(nn.Module):

  def __init__(self, n_features):
    super(Net, self).__init__()
    #Network
    if config.model == "SimpleNN":
        self.fc1 = nn.Linear(n_features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    #PhilipNetwork
    if config.model == "Philip":
        self.fc1 = nn.Linear(n_features, 1)


  def forward(self, x):
    if config.model == "SimpleNN":
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

    if config.model == "Philip":
        return torch.sigmoid(self.fc1(x))


def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)

def neuralnetwork(data, target):

    data = np.concatenate(data, axis=0 )
    target = np.concatenate(target, axis=0 )

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.squeeze(torch.from_numpy(y_train).float())
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.squeeze(torch.from_numpy(y_test).float())

    if config.model == "Philip":
        X_train = torch.unsqueeze(X_train, 1)
        #y_train = torch.unsqueeze(y_train, 1)
        X_test = torch.unsqueeze(X_test, 1)
        #y_test = torch.unsqueeze(y_test, 1)

    #test
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    if config.model == "SimpleNN":
        net = Net(X_train.shape[1])
    if config.model == "Philip":
        net = Net(n_features=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10000):

        y_pred = net(X_train)

        y_pred = torch.squeeze(y_pred)
        train_loss = criterion(y_pred, y_train)

        if epoch % 100 == 0:
            train_acc = calculate_accuracy(y_train, y_pred)

            y_test_pred = net(X_test)
            y_test_pred = torch.squeeze(y_test_pred)

            test_loss = criterion(y_test_pred, y_test)

            test_acc = calculate_accuracy(y_test, y_test_pred)
            print(f'''epoch {epoch}
            Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
            Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
            ''')

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    MODEL_PATH = 'model.pth'
    torch.save(net, MODEL_PATH)

    classes = ['No Optimum', 'Optimum']
    y_pred = net(X_test)
    y_pred = y_pred.ge(.5).view(-1).cpu()
    y_test = y_test.cpu()

    print(classification_report(y_test, y_pred, target_names=classes))

    return