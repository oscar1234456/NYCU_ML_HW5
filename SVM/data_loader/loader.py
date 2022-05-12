import numpy as np
import csv

PATH = "../data/"


def load_data(data_name):
    data_list = list()
    with open(PATH + data_name) as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            data_list.append(list(map(float, row)))
    data = np.array(data_list, dtype=np.double)
    return data


def load_all_data():
    return load_data("X_train.csv"), load_data("y_train.csv").reshape(-1), load_data("X_test.csv"), load_data(
        "y_test.csv").reshape(-1)
