import numpy as np
from libsvm.svmutil import *


def grid_search(log2c_parameters, log2g_parameters, kernel, training_data, X_test, y_test):
    # return grid_matrix, best parameters, best_parameters_testing_acc, best_training_acc
    best_training_acc = -999
    best_log2c_parameter = None
    best_log2g_parameter = None

    if kernel == "linear":
        kernel_index = 0
        log2g_parameters = [0]
    elif kernel == "polynomial":
        kernel_index = 1
    else:
        kernel_index = 2

    all_training_acc = np.zeros((len(log2c_parameters), len(log2g_parameters)))
    for now_focus_log2c in range(len(log2c_parameters)):
        for now_focus_log2g in range(len(log2g_parameters)):
            print(f"grid searching.... {log2c_parameters[now_focus_log2c], log2g_parameters[now_focus_log2g]}")
            parameter = f"-q -t {kernel_index} -v 2 -c {2 ** log2c_parameters[now_focus_log2c]}"
            if kernel_index != 0:
                parameter += f" -g {2 ** log2g_parameters[now_focus_log2g]}"
            print(parameter)
            result = svm_train(training_data, parameter)
            all_training_acc[now_focus_log2c, now_focus_log2g] = result
            if best_training_acc < result:
                best_training_acc = result
                best_log2c_parameter = log2c_parameters[now_focus_log2c]
                best_log2g_parameter = log2g_parameters[now_focus_log2g]
    # testing process
    parameter = f"-q -t {kernel_index} -c {2 ** best_log2c_parameter}"
    if kernel_index != 0:
        parameter += f" -g {2 ** best_log2g_parameter}"
    best_model = svm_train(training_data, parameter)
    p_labels, p_acc, p_vals = svm_predict(y_test, X_test, best_model)
    best_testing_acc = p_acc[0]
    return all_training_acc, best_log2c_parameter, best_log2g_parameter, best_training_acc, best_testing_acc
