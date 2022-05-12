from SVM.data_loader.loader import load_all_data
from libsvm.svmutil import *

# kernel argument
linear = "-t 0"
polynomial = "-t 1"
RBF = "-t 2"

# Data
X_train, y_train, X_test, y_test = load_all_data()
training_data = svm_problem(y_train, X_train)
testing_data = svm_problem(y_test, X_test)

# training linear
linear_svm = svm_train(training_data, "-q "+linear)
polynomial_svm = svm_train(training_data, "-q "+polynomial)
RBF_svm = svm_train(training_data, "-q "+RBF)

# testing
print("linear:")
svm_predict(y_test, X_test, linear_svm)
print("polynomial:")
svm_predict(y_test, X_test, polynomial_svm)
print("RBF:")
svm_predict(y_test, X_test, RBF_svm)

print()