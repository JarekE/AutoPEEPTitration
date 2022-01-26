# Simple NN
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import sys
import os
from torchmetrics import Accuracy
from sklearn.metrics import classification_report

import config


class Net(nn.Module):

  def __init__(self, n_features):
    super(Net, self).__init__()

    # Network
    if config.model == "SimpleNN":
        self.layers = nn.Sequential(
            nn.Linear(n_features, 4),
            nn.ReLU(),
            nn.Linear(4,6),
            nn.ReLU(),
            nn.Linear(6, 8),
            nn.ReLU(),
            nn.Linear(8, 10),
            nn.ReLU(),
            nn.Linear(10,12))
        self.fc1 = nn.Linear(12,1)

    # PhilipNetwork
    if config.model == "Philip":
        self.fc1 = nn.Linear(n_features, 1)


  def forward(self, x):
    if config.model == "SimpleNN":
        x = self.layers(x)
        return torch.sigmoid(self.fc1(x))

    if config.model == "Philip":
        return torch.sigmoid(self.fc1(x))


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


def loss_plot(train, test):

    plt.plot(train, '-b', label='train_loss')
    plt.plot(test, '-r', label='val_loss')

    plt.xlabel("n iteration")
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.savefig('loss_plot.png')
    plt.show()


def neuralnetwork(X_test, y_test, X_val, y_val, X_train, y_train):

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.squeeze(torch.from_numpy(y_train).float())
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.squeeze(torch.from_numpy(y_test).float())
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.squeeze(torch.from_numpy(y_val).float())

    if config.model == "Philip":
        X_train = torch.unsqueeze(X_train, 1)
        X_test = torch.unsqueeze(X_test, 1)
        X_val = torch.unsqueeze(X_val, 1)

    # test
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(X_val.shape, y_val.shape)

    # choose network
    if config.model == "SimpleNN":
        net = Net(X_train.shape[1])
    if config.model == "Philip":
        net = Net(n_features=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    # Metrics
    acc = Accuracy()

    if config.test == False:
        # training
        train = []
        val = []
        min_val = 1

        for epoch in range(config.epoch_number):

            # training
            y_pred = net(X_train)
            y_pred = torch.squeeze(y_pred)
            train_loss = criterion(y_pred, y_train)

            # validation
            y_val_pred = net(X_val)
            y_val_pred = torch.squeeze(y_val_pred)
            val_loss = criterion(y_val_pred, y_val)

            if config.loss_plot == True:
                train.append(train_loss)
                val.append(val_loss)

                if epoch % (config.epoch_number/10) == 0:
                    loss_plot(train, val)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_acc = acc(y_pred, y_train.int())
            val_acc = acc(y_val_pred, y_val.int())

            # Save the best model
            if round_tensor(val_loss) < min_val:

                MODEL_PATH = 'model_split' + str(config.split) + '.pth'
                torch.save(net, MODEL_PATH)

                min_val = round_tensor(val_loss)

            # Output
            if epoch % 100 == 0:

                print(f'''epoch {epoch}
                Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
                Val  set  - loss: {round_tensor(val_loss)}, accuracy: {round_tensor(val_acc)}
                ''')

            # Simple help against local minima
            if epoch % 100 == 0 and epoch >= (config.epoch_number-200) and round_tensor(train_acc) <= 0.94:
                print("Local minimum detected. Reset weights for optimal outcome.")
                os.system('python "main.py"')

        # testing (of best trained model)
        net = torch.load('model_split' + str(config.split) + '.pth')

        y_pred_test = net(X_test)
        y_pred_test = torch.squeeze(y_pred_test)
        test_acc = acc(y_pred_test, y_test.int())
        print(f'''Test set - accuracy: {round_tensor(test_acc)}''')

        classes = ['No Optimum', 'Optimum PEEP']
        y_pred = net(X_test)
        y_pred = y_pred.ge(.5).view(-1).cpu()
        y_test = y_test.cpu()

        if config.report == True:
            path = 'classification_report_split' + str(config.split) + '.txt'
            sys.stdout = open(path, "w")
            print(classification_report(y_test, y_pred, target_names=classes))
            sys.stdout.close()
        else:
            print(classification_report(y_test, y_pred, target_names=classes))

    else:

        net = torch.load('model_split' + str(config.split) + '.pth')

        y_pred_test = net(X_test)
        y_pred_test = torch.squeeze(y_pred_test)
        test_acc = acc(y_pred_test, y_test.int())

        print(f'''Test set - accuracy: {round_tensor(test_acc)}''')

        classes = ['No Optimum', 'Optimum PEEP']
        y_pred = net(X_test)
        y_pred = y_pred.ge(.5).view(-1).cpu()
        y_test = y_test.cpu()

        print(classification_report(y_test, y_pred, target_names=classes))

    return