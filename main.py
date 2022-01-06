#Project from Ines and Jarek

from Dataloader import loader
import config
from SimpleNN import neuralnetwork
from SVMmodel import svm
from KNNmodel import kNN
from Compare_ML_Classifiers import comp_clf

def main():

    data_list, target_list = loader()

    if config.compare == True:
        comp_clf(data_list, target_list)
        return

    if config.model == "SimpleNN":
        neuralnetwork(data_list, target_list)
    elif config.model == "Philip":
        neuralnetwork(data_list, target_list)
    elif config.model == "SVM":
        svm(data_list, target_list)
    elif config.model == "kNN":
        kNN(data_list, target_list)
    else:
        raise Exception("Wrong model")

    return

if __name__ == '__main__':

    main()