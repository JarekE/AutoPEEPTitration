# Load the data
import pandas as pd
import os
from os import listdir
import config
import numpy as np

"""

Idee:
2 Ansätze: Einmal mit allen Daten und einem NN, das andere ein RNN mit den C Daten

"""


def create_data(set, name):
    # how many breathes per peep step
    length = config.length
    # control variable
    name = name

    if config.model == "RNN":
        data_set = set[["p_peep", "C_rs_est"]].to_numpy()
        width = 2
    # TODO: Entscheiden welche Merkmale wirklich wichtig sein könnten (zB durch KNN etc) --> Kann noch warten!
    elif config.model == "SimpleNN":
        data_set = set[["p_peep", "p_pip", "p_plat", "P_et_CO2", "R_rs_est", "C_rs_est"]].to_numpy()
        width = 6

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