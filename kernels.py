import numpy as np


def linear_kernel(X_left, X_right):
    """ 
    The gram matrix for a linear kernel
    
    Parameters
    ----------
        X_left : np.array of shape (n_left, m)
        X_right : np.array of shape (n_right, m)

    Returns
    -------
        K : gram matrix of shape (n_left, n_right)
            K_ij = Inner_Product(X_left_i, X_right_j)
    """
    return X_left @ X_right.T

#===============================================================

def gaussian_kernel(X_left, X_right, sigma2=1):
    """ 
    The gram matrix for a gaussian kernel
    
    Parameters
    ----------
        X_left : np.array of shape (n_left, m)
        X_right : np.array of shape (n_right, m)

    Returns
    -------
        K : gram matrix of shape (n_left, n_right)
            K_ij = Gauss_Kernel(X_left_i, X_right_j)
    """
    diff = (X_left[:, None, :] - X_right[None, : , :])**2
    norm2 = diff.sum(axis=2)
    return np.exp(-norm2/sigma2)

#===============================================================

def k_substrings_embedding(s, k=3):
    """
    computes an embedding for a given string s
    number of occurences of all k substrings of s

    Parameters
    ----------
        s : str
            input string
        k : int
            number of charaters to consider

    Returns
    -------
        Phi : dict (spectrum embedding of s)
            the keys are k substrings of s
            the values are the number of occurences of the keys in s
    """
    phi = {}
    for i in range(len(s)-k+1):
        sub = s[i:i+k]
        if sub in phi:
            phi[sub] += 1
        else:
            phi[sub] = 1
    return phi
   
def seq_inner_product(phi1, phi2):
    """
    computes the inner product between two given 
    spectrum embeddings

    Parameters
    ----------
        phi1 : dict
            a spectrum embeddings
        phi2 : dict
            a spectrum embeddings

    Returns
    -------
        p : int
            p = sum_a phi1(a)*phi2(a) 
            where a is a commun substring of phi1 and phi2
    """
    p = 0
    for sub in phi1:
        if sub in phi2:
            p += phi1[sub]*phi2[sub]    
    return p

def spectrum_kernel(X_left, X_right):
    """ 
    The gram matrix for the spectrum kernel
    
    Parameters
    ----------
        X_left : np.array of n_left dicts (spectrum embeddings)
        X_right : np.array of n_right dicts (spectrum embeddings)

    Returns
    -------
        K : gram matrix of shape (n_left, n_right)
            K_ij = Spectrum_Kernel(X_left_i, X_right_j)
    """
    n1 = X_left.shape[0]
    n2 = X_right.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i,j] = seq_inner_product(X_left[i], X_right[j])
            
    return K