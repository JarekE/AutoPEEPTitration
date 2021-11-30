# Load the data
import pandas as pd
import os
from os import listdir
import numpy as np
import config
import matplotlib.pyplot as plt


def gradient_c(median_step):

    length = len(median_step) * config.length
    grad_array = np.zeros(length)

    for x in range(1,len(median_step)):
        grad_array[x*config.length:x*config.length+config.length] = (median_step[x]-median_step[x-1])/median_step[x-1]
    grad_array[0:config.length] = grad_array[config.length:2*config.length]

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
        data_set = set[["p_peep", "C_rs_eve", "C_rs_est"]].to_numpy()
        width = 3

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
            # calculate the median c_rs_est per step
            median_step = []
            for step in range(len(list)):
                calc = np.median(list[step][:, -1])
                median_step.append(calc)

            #Now the gradient is the last datapoint
            data = np.concatenate((data, gradient_c(median_step)), axis=1)

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
        path = os.path.join("breath_data_optimal", name)
        data, target = create_data(pd.read_csv(filepath_or_buffer=path), name)
        data_list.append(data)
        target_list.append(target)

    #My little plot function -> more information about each graph
    if config.print == True:
        for i in range(len(data_list)):
            plt.figure(i + 1)
            if config.model == "RNN" or config.model == "LSTM":
                [a, b] = plt.plot(data_list[i])
                [c] = plt.plot(target_list[i])
                plt.legend([a, b, c], ['p_peep', 'C_rs_eve', 'one_hot_encoding'])
            elif config.model == "SimpleNN":
                [a, b, c, d] = plt.plot(data_list[i])
                [e] = plt.plot(target_list[i])
                plt.legend([a, b, c, d, e], ['p_peep', 'C_rs_eve', 'C_rs_est', 'gradient', 'one_hot_encoding'])
            elif config.model == "Philip":
                [a] = plt.plot(data_list[i])
                [b] = plt.plot(target_list[i])
                plt.legend([a, b], ['gradient', 'one_hot_encoding'])

            plt.show()

    return data_list, target_list