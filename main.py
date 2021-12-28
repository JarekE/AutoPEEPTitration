#Project from Ines and Jarek

from Dataloader import loader
import config
from SimpleNN import neuralnetwork

def main():

    data_list, target_list = loader()

    if config.model == "SimpleNN":
        neuralnetwork(data_list, target_list)
    elif config.model == "Philip":
        neuralnetwork(data_list, target_list)
    else:
        raise Exception("Wrong model")

    return

if __name__ == '__main__':

    main()


