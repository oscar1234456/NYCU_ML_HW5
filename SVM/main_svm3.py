import pickle

import numpy as np
from libsvm.svmutil import *

from scipy.spatial.distance import cdist

from SVM.data_loader.loader import load_all_data
from SVM.plot.heat_map import pretty_plot
from SVM.search.grid_search import grid_search, grid_search_custom





# Data
X_train, y_train, X_test, y_test = load_all_data()



print("_____linear + RBF_____")
kernel = "RBF"
log2c_parameters = [i - 8 for i in range(0, 18, 2)]
log2g_parameters = [i - 8 for i in range(0, 18, 2)]
# log2c_parameters = [1]
# log2g_parameters = [4]
all_training_acc, best_log2c_parameter, best_log2g_parameter, best_training_acc, best_testing_acc = \
    grid_search_custom(log2c_parameters, log2g_parameters, X_train, y_train, X_test, y_test)
print("_____linear + RBF_____")
print(f"Best log2c: {best_log2c_parameter}")
print(f"Best log2g: {best_log2g_parameter}")
print(f"Best Training Accuracy: {best_training_acc}")
print(f"Testing Accuracy: {best_testing_acc}")
print(f"All Training Accuracy: {all_training_acc}")
with open('all_acc_combine.pickle', 'wb') as f:
    pickle.dump(all_training_acc, f)
pretty_plot('all_acc_combine.pickle', log2c_parameters, log2g_parameters, "linear + RBF")
print()


