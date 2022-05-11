import numpy as np
import matplotlib.pyplot as plt

from GP.dataloader.loader import data_loader

# Get data
from GP.kernel.gaussian_process import gaussian_process
from GP.kernel.kernel import kernel

X, y = data_loader()

# parameters
sigma_square = 1
alpha = 1
length_scale = 1
beta = 5  # from homework notice

# covariance matrix
k = kernel(X, X.T, sigma_square=sigma_square, alpha=alpha, length_scale=length_scale)
C = k + (1 / beta) * np.identity(X.shape[0])  # beta[scalar].inv = reciprocal

x_star = np.linspace(-60, 60, num=450).reshape(-1, 1)

# gaussian_process(x, x_star, C, y, sigma_square=1, alpha=1, length_scale=1)
predict_mean, predict_var = gaussian_process(X, x_star, C, y, sigma_square=sigma_square, alpha=alpha,
                                             length_scale=length_scale)

predict_std = np.sqrt(np.diag(predict_var))

plt.plot(X, y, "bo")
plt.plot(x_star, predict_mean, 'k-')
plt.fill_between(x_star.reshape(-1), predict_mean.reshape(-1) + 1.96 * predict_std,
                 predict_mean.reshape(-1) - 1.96 * predict_std, facecolor='salmon')
plt.xlim(-60,60)
plt.show()
