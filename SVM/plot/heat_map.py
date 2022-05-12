import pickle

import numpy as np
import matplotlib.pyplot as plt


def pretty_plot(pickle_name, log2c_parameters, log2g_parameters, kernel):
    # Load
    with open(f'{pickle_name}', 'rb') as f:
        acc_matrix = pickle.load(f)
    print(acc_matrix)
    plt.figure(figsize=(8, 6))

    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(acc_matrix.T, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xlabel('log2c')
    plt.ylabel('log2g')
    plt.colorbar()

    plt.yticks(np.arange(len(log2g_parameters)), log2g_parameters)
    plt.xticks(np.arange(len(log2c_parameters)), log2c_parameters)

    for i in range(acc_matrix.shape[0]):
        for j in range(acc_matrix.shape[1]):
            print(f"{i, j}")
            plt.text(i, j, round(acc_matrix[i, j], 2), ha='center', va="center", color="green", weight="bold")

    plt.title(f'Grid Search ({kernel})')
    if kernel == "linear":
        ax = plt.gca()
        ax.axes.yaxis.set_visible(False)
    plt.show()
