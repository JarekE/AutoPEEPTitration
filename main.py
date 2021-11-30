#Project from Ines and Jarek
from Dataloader import loader
from RNNModel import rnn
import config
#from SimpleNN import neuralnetwork
from LSTMmodel import lstm

def main():

    data_list, target_list = loader()

    if config.model == "RNN":
        rnn(data_list, target_list)
    elif config.model == "SimpleNN":
        neuralnetwork(data_list, target_list)
    elif config.model == "Philip":
        #neuralnetwork(data_list, target_list)
        print(test)
    elif config.model == "LSTM":
        lstm(data_list, target_list)
    else:
        raise Exception("Wrong model")

    return

# Testcomment123
if __name__ == '__main__':

    main()


