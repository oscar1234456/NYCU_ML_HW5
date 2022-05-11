import numpy as np

from GP.kernel.kernel import kernel


def gaussian_process(x, x_star, C, y, sigma_square=1, alpha=1, length_scale=1):
    # parameters
    beta = 5  # from homework notice

    kernel_x_x_star = kernel(x, x_star.T, sigma_square=sigma_square, alpha=alpha,
                             length_scale=length_scale)
    kernel_x_star_x_star = kernel(x_star, x_star.T, sigma_square=sigma_square, alpha=alpha,
                                  length_scale=length_scale)
    k_star = kernel_x_star_x_star + (1 / beta) * np.identity(kernel_x_star_x_star.shape[0])

    predict_mean = kernel_x_x_star.T @ np.linalg.inv(C) @ y
    predict_var = k_star - kernel_x_x_star.T @ np.linalg.inv(C) @ kernel_x_x_star

    return predict_mean, predict_var
