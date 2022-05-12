import pickle

import numpy as np
from libsvm.svmutil import *

from scipy.spatial.distance import cdist

from SVM.data_loader.loader import load_all_data
from SVM.search.grid_search import grid_search, grid_search_custom


def custom_kernel(X_1, X_2, g):
    linear_kernel = X_1 @ X_2.T
    rbf_kernel = np.exp(-g * cdist(X_1, X_2, 'speuclidean'))
    combine_kernel = linear_kernel + rbf_kernel
    index_vector = np.arange(1, len(X_1) + 1)
    return np.concatenate([index_vector, combine_kernel], axis=1)


# Data
X_train, y_train, X_test, y_test = load_all_data()
custom_kernel(X_train, X_train, )
training_data = svm_problem(y_train, X_train)




print("_____Radial Basis Function_____")
kernel = "RBF"
log2c_parameters = [i - 8 for i in range(0, 18, 2)]
log2g_parameters = [i - 8 for i in range(0, 18, 2)]
# log2c_parameters = [1,2]
# log2g_parameters = [4,5]
all_training_acc, best_log2c_parameter, best_log2g_parameter, best_training_acc, best_testing_acc = \
    grid_search_custom(log2c_parameters, log2g_parameters, kernel_test, training_data, X_test, y_test)
print("_____Radial Basis Function Result_____")
print(f"Best log2c: {best_log2c_parameter}")
print(f"Best log2g: {best_log2g_parameter}")
print(f"Best Training Accuracy: {best_training_acc}")
print(f"Testing Accuracy: {best_testing_acc}")
print(f"All Training Accuracy: {all_training_acc}")
with open('all_acc_combine.pickle', 'wb') as f:
    pickle.dump(all_training_acc, f)



