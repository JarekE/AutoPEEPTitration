#Project from Ines and Jarek
from Dataloader import loader
from RNNModel import rnn

def main():

    data_list = loader()
    rnn(data_list)

    return

if __name__ == '__main__':

    main()


