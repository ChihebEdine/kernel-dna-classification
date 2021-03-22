import numpy as np


def linear_kernel(X_left, X_right):
    return X_left @ X_right.T

#===============================================================

def gaussian_kernel(X_left, X_right, sigma2=1):
    diff = (X_left[:, None, :] - X_right[None, : , :])**2
    norm2 = diff.sum(axis=2)
    return np.exp(-norm2/sigma2)

#===============================================================

def k_substrings_embedding(s, k=3):
    phi = {}
    for i in range(len(s)-k+1):
        sub = s[i:i+k]
        if sub in phi:
            phi[sub] += 1
        else:
            phi[sub] = 1
    return phi
   
def seq_inner_product(phi1, phi2):
    p = 0
    for sub in phi1:
        if sub in phi2:
            p += phi1[sub]*phi2[sub]    
    return p

def spectrum_kernel(X_left, X_right):
    n1 = X_left.shape[0]
    n2 = X_right.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i,j] = seq_inner_product(X_left[i], X_right[j])
            
    return K