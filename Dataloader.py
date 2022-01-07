# Load the data
import pandas as pd
import os
from os import listdir
import numpy as np
import config
import matplotlib.pyplot as plt


# calculate the gradient of the compliance
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

    # not in use at the moment
    if config.model == "RNN":
        data_set = set[["p_peep", "C_rs_est"]].to_numpy()
        width = 2
    elif config.model == "SimpleNN" or config.model == "Philip" or config.model == "kNN" or config.model == "SVM":
        #C_rs_est has to be the last datapoint! (--> gradient_c() )
        if config.compare == True or config.model == "SVM":
            data_set = set[["p_peep", "C_rs_est"]].to_numpy()
            width = 2
        else:
            data_set = set[["p_peep", "C_rs_eve", "C_rs_est"]].to_numpy()
            width = 3

        # since all stairs begin with peep = 24 --> can be generalized if necessary
        peep = 23
        list = []

        # peep of the last step
        minimal_peep = round(sum(data_set[-10:-1, 0]/9))
        if minimal_peep == 7:
            minimal_peep = 6
        if minimal_peep == 13:
            minimal_peep = 12

        # total number of steps
        number_of_steps = int(((24 - minimal_peep) / 2 + 1))

        print(minimal_peep)
        print(name)

        while peep >= (minimal_peep-1):
            data = data_set[(data_set[..., 0] > peep) & (data_set[..., 0] < (peep+2))].astype(np.float16)
            array = np.empty([length, width], np.float16)
            for x in range(0, length):
                array[x, :] = data[x, :]
            list.append(array)
            peep = peep-2

        data = np.vstack(list)
        target = target_vector(list, length, number_of_steps)

        if config.grad == True and config.compare == False:
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


def target_vector(list, length, number):

    target_list = []
    for x in range(0,number):
        average = np.average(list[x][..., -1])
        target_list.append(average)

    max_value = max(target_list)
    max_index = target_list.index(max_value)

    target = np.zeros((length*number, 1))
    target[(max_index*length):(max_index*length+length)] = np.ones((length, 1))

    return target


def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def loader():

    data_list = list()
    target_list = list()
    filenames = find_csv_filenames("breath_data")

    for name in filenames:
        path = os.path.join("breath_data", name)
        data, target = create_data(pd.read_csv(filepath_or_buffer=path), name)
        data_list.append(data)
        target_list.append(target)

    #plot function -> more information and visualization about each graph
    if config.print == True:
        for i in range(len(data_list)):
            plt.figure(i + 1)
            if config.model == "SimpleNN":
                [a, b, c, d] = plt.plot(data_list[i])
                [e] = plt.plot(target_list[i])
                plt.legend([a, b, c, d, e], ['p_peep', 'C_rs_eve', 'C_rs_est', 'gradient', 'one_hot_encoding'])
            elif config.model == "Philip":
                [a] = plt.plot(data_list[i])
                [b] = plt.plot(target_list[i])
                plt.legend([a, b], ['gradient', 'one_hot_encoding'])

            #plt.savefig('plot'+str(i+1)+'.png') if you want to save the plots
            plt.show()

    return data_list, target_list