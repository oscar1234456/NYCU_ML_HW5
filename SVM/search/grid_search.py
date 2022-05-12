import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import cdist


def custom_kernel(X_1, X_2, g):
    linear_kernel = X_1 @ X_2.T
    rbf_kernel = np.exp(-g * cdist(X_1, X_2, 'sqeuclidean'))
    combine_kernel = linear_kernel + rbf_kernel
    index_vector = np.arange(1, len(X_1) + 1).reshape(-1, 1)
    return np.concatenate([index_vector, combine_kernel], axis=1)


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
            parameter = f"-q -t {kernel_index} -v 3 -c {2 ** log2c_parameters[now_focus_log2c]}"
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


def grid_search_custom(log2c_parameters, log2g_parameters, X_train, y_train, X_test, y_test):
    # return grid_matrix, best parameters, best_parameters_testing_acc, best_training_acc
    best_training_acc = -999
    best_log2c_parameter = None
    best_log2g_parameter = None

    all_training_acc = np.zeros((len(log2c_parameters), len(log2g_parameters)))
    for now_focus_log2c in range(len(log2c_parameters)):
        for now_focus_log2g in range(len(log2g_parameters)):
            print(f"grid searching.... {log2c_parameters[now_focus_log2c], log2g_parameters[now_focus_log2g]}")
            parameter = f"-q -t 4 -v 3 -c {2 ** log2c_parameters[now_focus_log2c]} -g {2 ** log2g_parameters[now_focus_log2g]}"
            print(parameter)
            new_kernel_train = custom_kernel(X_train, X_train, 2 ** log2g_parameters[now_focus_log2g])
            training_data = svm_problem(y_train, new_kernel_train, isKernel=True)
            result = svm_train(training_data, parameter)
            all_training_acc[now_focus_log2c, now_focus_log2g] = result
            if best_training_acc < result:
                best_training_acc = result
                best_log2c_parameter = log2c_parameters[now_focus_log2c]
                best_log2g_parameter = log2g_parameters[now_focus_log2g]
    # testing process
    parameter = f"-q -t 4 -c {2 ** best_log2c_parameter} -g {2 ** best_log2g_parameter}"

    new_kernel_train_testing = custom_kernel(X_train, X_train, 2 ** best_log2g_parameter)
    train_testing_data = svm_problem(y_train, new_kernel_train_testing, isKernel=True)
    best_model = svm_train(train_testing_data, parameter)

    new_kernel_test = custom_kernel(X_test, X_train, 2 ** best_log2g_parameter)
    p_labels, p_acc, p_vals = svm_predict(y_test, new_kernel_test, best_model)
    best_testing_acc = p_acc[0]
    return all_training_acc, best_log2c_parameter, best_log2g_parameter, best_training_acc, best_testing_acc
