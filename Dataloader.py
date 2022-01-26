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

    if config.model == "SimpleNN" or config.model == "Philip" or config.model == "kNN" or config.model == "SVM":
        #C_rs_est has to be the last datapoint! (--> gradient_c() )
        if config.compare == True or config.model == "SVM":
            data_set = set[["p_peep", "C_rs_est"]].to_numpy()
            width = 2
        else:
            data_set = set[["p_peep", "C_rs_est"]].to_numpy()
            width = 2

        # since all stairs begin with peep = 24 --> can be generalized if necessary
        peep = 23
        list = []

        # peep of the last step (with odd values)
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

            if config.no_peep:
                data = data[:, 1:width+1]

        if config.equal_size:
            # find data points with maximal compliance -> 1 in target vector
            index = np.where(target == 1)
            max_compliance_data = np.take(data, index[0], axis=0)
            one_vector = np.take(target, index[0], axis=0)

            # add this data to the vector until we reach a equal number of 1 and 0 in the target vector
            for x in range(number_of_steps-2):
                data = np.append(data, max_compliance_data, axis = 0)
                target = np.append(target, one_vector, axis = 0)

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

    # 1-hot-encoding of the maximum vector
    target = np.zeros((length*number, 1))
    target[(max_index*length):(max_index*length+length)] = np.ones((length, 1))

    # 1-hot-encoding of the not-the-maximum vector
    """
    if config.model == "SimpleNN" or config.model == "Philip":
        target2 = np.ones((length * number, 1))
        target2[(max_index * length):(max_index * length + length)] = np.zeros((length, 1))

        target = np.concatenate((target, target2), axis=1)
    """

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

    if config.split == 1:
        X_test = data_list[0:4]
        y_test = target_list[0:4]
        X_val = data_list[4:8]
        y_val = target_list[4:8]
        X_train = data_list[8:28]
        y_train = target_list[8:28]
    elif config.split == 2:
        X_test = data_list[4:8]
        y_test = target_list[4:8]
        X_val = data_list[8:12]
        y_val = target_list[8:12]
        X_train = data_list[0:4]+data_list[12:28]
        y_train = target_list[0:4]+target_list[12:28]
    elif config.split == 3:
        X_test = data_list[8:12]
        y_test = target_list[8:12]
        X_val = data_list[12:16]
        y_val = target_list[12:16]
        X_train = data_list[0:8]+data_list[16:28]
        y_train = target_list[0:8]+target_list[16:28]
    elif config.split == 4:
        X_test = data_list[12:16]
        y_test = target_list[12:16]
        X_val = data_list[16:20]
        y_val = target_list[16:20]
        X_train = data_list[0:12]+data_list[20:28]
        y_train = target_list[0:12]+target_list[20:28]
    elif config.split == 5:
        X_test = data_list[16:20]
        y_test = target_list[16:20]
        X_val = data_list[20:24]
        y_val = target_list[20:24]
        X_train = data_list[0:16]+data_list[24:28]
        y_train = target_list[0:16]+target_list[24:28]
    elif config.split == 6:
        X_test = data_list[20:24]
        y_test = target_list[20:24]
        X_val = data_list[24:28]
        y_val = target_list[24:28]
        X_train = data_list[0:20]
        y_train = target_list[0:20]
    elif config.split == 7:
        X_test = data_list[24:28]
        y_test = target_list[24:28]
        X_val = data_list[0:4]
        y_val = target_list[0:4]
        X_train = data_list[4:24]
        y_train = target_list[4:24]
    else:
        raise Exception("Choose a number between 1-7 for the split")

    # data_list to numpy
    X_test = np.concatenate(X_test, axis=0 )
    y_test = np.concatenate(y_test, axis=0 )
    X_val = np.concatenate(X_val, axis=0 )
    y_val = np.concatenate(y_val, axis=0 )
    X_train = np.concatenate(X_train, axis=0 )
    y_train = np.concatenate(y_train, axis=0 )

    return X_test, y_test, X_val, y_val, X_train, y_train