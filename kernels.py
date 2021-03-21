import numpy as np


def linear_kernel(X_left, X_right):
    return X_left @ X_right.T

def gaussian_kernel(X_left, X_right, sigma2=1):
    diff = (X_left[:, None, :] - X_right[None, : , :])**2
    norm2 = diff.sum(axis=2)
    return np.exp(-norm2/sigma2)