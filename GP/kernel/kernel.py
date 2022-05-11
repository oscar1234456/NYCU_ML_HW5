import numpy as np


def kernel(X1, X2, sigma_square=1, alpha=1, length_scale=1):
    # Rational quadratic kernel
    # k(x1, x2) = sigma_square^2 * [1+ (x1-x2)^2/2 * alpha * (length_scale^2)] ^ (-alpha)
    X1_minus_X2_square = np.power(X1 - X2, 2)
    return sigma_square * (1 + (X1_minus_X2_square / (2 * alpha * length_scale ** 2))) ** (-alpha)
