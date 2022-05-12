import pickle

from SVM.data_loader.loader import load_all_data
from libsvm.svmutil import *

# kernel argument
from SVM.plot.heat_map import pretty_plot
from SVM.search.grid_search import grid_search

# Data
X_train, y_train, X_test, y_test = load_all_data()
training_data = svm_problem(y_train, X_train)
testing_data = svm_problem(y_test, X_test)

# grid search
# grid_search(log2c_parameters, log2g_parameters, kernel, training_data, X_test, y_test)
# log2c_parameters = [i-8 for i in range(0, 18, 2)]
# log2g_parameters = [i-8 for i in range(0, 18, 2)]


# print("_____Linear_____")
# kernel = "linear"
# log2c_parameters = [i - 8 for i in range(0, 18, 2)]
# log2g_parameters = [0]
# all_training_acc, best_log2c_parameter, best_log2g_parameter, best_training_acc, best_testing_acc = \
#     grid_search(log2c_parameters, log2g_parameters, kernel, training_data, X_test, y_test)
# print("_____Linear Result_____")
# print(f"Best log2c: {best_log2c_parameter}")
# print(f"Best Training Accuracy: {best_training_acc}")
# print(f"Testing Accuracy: {best_testing_acc}")
# print(f"All Training Accuracy: {all_training_acc}")
# with open('all_acc_linear.pickle', 'wb') as f:
#     pickle.dump(all_training_acc, f)
# pretty_plot('all_acc_linear.pickle', log2c_parameters, log2g_parameters, "linear")
# print()

print("_____Polynomial_____")
kernel = "polynomial"
# log2c_parameters = [i - 8 for i in range(0, 18, 2)]
# log2g_parameters = [i - 8 for i in range(0, 18, 2)]
log2c_parameters = [1,2,3]
log2g_parameters = [4,5,6]
all_training_acc, best_log2c_parameter, best_log2g_parameter, best_training_acc, best_testing_acc = \
    grid_search(log2c_parameters, log2g_parameters, kernel, training_data, X_test, y_test)
print("_____Polynomial Result_____")
print(f"Best log2c: {best_log2c_parameter}")
print(f"Best log2g: {best_log2g_parameter}")
print(f"Best Training Accuracy: {best_training_acc}")
print(f"Testing Accuracy: {best_testing_acc}")
print(f"All Training Accuracy: {all_training_acc}")
with open('all_acc_polynomial.pickle', 'wb') as f:
    pickle.dump(all_training_acc, f)
pretty_plot('all_acc_polynomial.pickle', log2c_parameters, log2g_parameters, "polynomial")
print()
