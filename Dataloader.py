# Load the data
import pandas as pd
import os
from os import listdir

# TODO: Jeden Datensatz so bereinigen, dass nur noch die Daten gespeichert sind, welche wir brauchen
# zB nur noch die ersten 20 Atemzüge eines PEEP-Sprungs mit den C_rs_est Daten (oder was halt sinnvoll ist)
def clear_data(set):

    data_set = set[set.columns[::16]]

    return data_set

#TODO: Funktion sollte zu jedem Datensatz den "Lösungsdatensatz" (= Target) in Form eines 1-hot-encoding vectors beinhalten
def target_vector(set):

    data_set = set

    return data_set


def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def loader():

    data_list = list()
    filenames = find_csv_filenames("breath_data")

    for name in filenames:
        path = os.path.join("breath_data", name)
        data_set = clear_data(pd.read_csv(filepath_or_buffer=path))
        data_set = target_vector(data_set)
        data_list.append(data_set)

    return data_list
