import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from GP.dataloader.loader import data_loader

# Get data
from GP.kernel.gaussian_process import gaussian_process
from GP.kernel.kernel import kernel


def objective_function(parameters, X, beta, y):
    parameters = parameters.reshape(-1)
    k = kernel(X, X.T, sigma_square=parameters[0], alpha=parameters[1], length_scale=parameters[2])
    C = k + (1 / beta) * np.identity(X.shape[0])
    obj = 0.5 * np.log(np.linalg.det(C)) + 0.5 * (y.T @ np.linalg.inv(C) @ y) \
          + 0.5 * X.shape[0] * np.log(2 * np.pi)
    return obj.reshape(-1)


X, y = data_loader()

# parameters
sigma_square = 1
alpha = 1
length_scale = 1
beta = 5  # from homework notice

# covariance matrix
k = kernel(X, X.T, sigma_square=sigma_square, alpha=alpha, length_scale=length_scale)
C = k + (1 / beta) * np.identity(X.shape[0])  # beta[scalar].inv = reciprocal

# opt
optimal_parameters = minimize(objective_function, [sigma_square, alpha, length_scale],
                              bounds=((1e-8, 1e6), (1e-8, 1e6), (1e-8, 1e6)),
                              args=(X, beta, y))
optimal_sigma_square = optimal_parameters.x[0]
optimal_alpha = optimal_parameters.x[1]
optimal_length_scale = optimal_parameters.x[2]

k_optimal = kernel(X, X.T, sigma_square=optimal_sigma_square, alpha=optimal_alpha,
                   length_scale=optimal_length_scale)
C_optimal = k_optimal + (1 / beta) * np.identity(X.shape[0])  # beta[scalar].inv = reciprocal

x_star = np.linspace(-60, 60, num=450).reshape(-1, 1)

# gaussian_process(x, x_star, C, y, sigma_square=1, alpha=1, length_scale=1)
predict_mean, predict_var = gaussian_process(X, x_star, C, y, sigma_square=sigma_square, alpha=alpha,
                                             length_scale=length_scale)

predict_std = np.sqrt(np.diag(predict_var))

# gaussian_process(x, x_star, C, y, sigma_square=1, alpha=1, length_scale=1)
predict_mean_optimal, predict_var_optimal = gaussian_process(X, x_star, C_optimal, y, sigma_square=optimal_sigma_square,
                                                             alpha=optimal_alpha,
                                                             length_scale=optimal_length_scale)

predict_std_optimal = np.sqrt(np.diag(predict_var_optimal))

# plot
f, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(X, y, "bo")
axs[0].plot(x_star, predict_mean, 'k-')
axs[0].fill_between(x_star.reshape(-1), predict_mean.reshape(-1) + 1.96 * predict_std,
                    predict_mean.reshape(-1) - 1.96 * predict_std, facecolor='salmon')
axs[0].set_xlim(-60, 60)

axs[1].plot(X, y, "bo")
axs[1].plot(x_star, predict_mean_optimal, 'k-')
axs[1].fill_between(x_star.reshape(-1), predict_mean_optimal.reshape(-1) + 1.96 * predict_std_optimal,
                    predict_mean_optimal.reshape(-1) - 1.96 * predict_std_optimal, facecolor='salmon')
axs[1].set_xlim(-60, 60)
plt.show()
