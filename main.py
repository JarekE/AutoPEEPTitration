#Project from Ines and Jarek

from Dataloader import loader
import config
from SimpleNN import neuralnetwork
from SVMmodel import svm
from KNNmodel import kNN
from Compare_ML_Classifiers import comp_clf


def main():

    # load data
    X_test, y_test, X_val, y_val, X_train, y_train = loader()

    # use data
    if config.compare == True:
        comp_clf(X_test, y_test, X_val, y_val, X_train, y_train)
        return

    if config.model == "SimpleNN":
        neuralnetwork(X_test, y_test, X_val, y_val, X_train, y_train)
    elif config.model == "Philip":
        neuralnetwork(X_test, y_test, X_val, y_val, X_train, y_train)
    elif config.model == "SVM":
        svm(X_test, y_test, X_val, y_val, X_train, y_train)
    elif config.model == "kNN":
        kNN(X_test, y_test, X_val, y_val, X_train, y_train)
    else:
        raise Exception("Wrong model")

    return


if __name__ == '__main__':

    main()