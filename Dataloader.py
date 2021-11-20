# Load the data
import pandas as pd
import os
from os import listdir
import numpy as np
import config

"""

Idee:
2 Ansätze: Einmal mit allen Daten und einem NN, das andere ein RNN mit den C Daten

"""


def gradient_c(data):

    #Array of C_rs_est
    c_array = data[:,-1]

    grad_array = np.subtract(c_array[1:], c_array[0:-1]) / c_array[0:-1]
    #Otherwith wrong length
    grad_array = np.append(grad_array, grad_array[-1])

    return np.expand_dims(grad_array, axis=1)


def create_data(set, name):
    # how many breathes per peep step
    length = config.length
    # control variable
    name = name

    if config.model == "RNN":
        data_set = set[["p_peep", "C_rs_est"]].to_numpy()
        width = 2

    # TODO: Entscheiden welche Merkmale wirklich wichtig sein könnten (zB durch KNN etc) --> Kann noch warten!
    elif config.model == "SimpleNN" or config.model == "Philip":
        #C_rs_est has to be the last datapoint! (--> gradient_c() )
        data_set = set[["p_peep", "C_rs_eve", "R_rs_eve", "R_rs_est", "C_rs_est"]].to_numpy()
        width = 5

    #numpy arrays with the number of breathes X per peep step
        peep = 23
        list = []
        while peep >= 5:
            data = data_set[(data_set[..., 0] > peep) & (data_set[..., 0] < (peep+2))].astype(np.float16)
            array = np.empty([length, width], np.float16)
            for x in range(0, length):
                array[x, :] = data[x, :]
            list.append(array)
            peep = peep-2

        target = target_vector(list, length, width)
        data = np.concatenate((list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7], list[8], list[9]))

        if config.grad:
            #Now, the gradient is the last datapoint
            data = np.concatenate((data, gradient_c(data)), axis=1)

        if config.model == "Philip":
            if config.grad != True:
                raise Exception("Turn On (True) the gradient in .config")
            data = data[:, -1]

    return data, target


def target_vector(list, length, width):

    target_list = []
    for x in range(0,10):
        average = np.average(list[x][..., -1])
        target_list.append(average)

    max_value = max(target_list)
    max_index = target_list.index(max_value)

    target = np.zeros((length*10, 1))
    target[(max_index*length):(max_index*length+length)] = np.ones((length, 1))

    return target


def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

#TODO: WICHTIG: Checken ob alle Datensätze (28) wie gewollt gespeichert werden und vor allem das target stimmt (Fleißarbeit!! am besten mit visualisierung)
def loader():

    data_list = list()
    target_list = list()
    filenames = find_csv_filenames("breath_data_optimal")

    for name in filenames:
        path = os.path.join("breath_data", name)
        data, target = create_data(pd.read_csv(filepath_or_buffer=path), name)
        data_list.append(data)
        target_list.append(target)

    return data_list, target_list