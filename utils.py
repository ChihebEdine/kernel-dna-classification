import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from models import KernelRidgeRegression, KernelLogisticRegression, KernelSVM
from kernels import linear_kernel, gaussian_kernel, k_substrings_embedding, spectrum_kernel


def accuracy(y_test, y_pred):
    """
    accuracy between preditions and ground truth

    Parameters
    ----------
        y_test : np.array of int with shape (n,)
            ground truth labels 
        y_pred : np.array of int with shape (n,)
            predicted labels
        
    Returns
    -------
        float : accuracy score
    """
    return (y_pred == y_test).astype(int).mean()


def best_threshold(model, X_test, y_test, lb=-0.5, ub=0.5, precision=20, plot=True):
    """
    search for best prediction threshold for a model

    Parameters
    ----------
        model : class defined in models.py
        X_test : features 
        y_test : np.array of int with shape (n,)
            ground truth labels 
        lb : float
            lower bound for threshold search
        ub : float
            upper bound for threshold search
        precision : int
            number of thresholds to test in the interval [lb, ub]
        plot : bool
            if True plots the accuracy w.r.t the threshold
    Returns
    -------
        float : accuracy score
    """
    thresholds = np.linspace(lb, ub, precision)
    accuracies = []
    print("Looking for best thresholds ...")
    for threshold in tqdm(thresholds):
        y_pred, _ = model.predict(X_test, threshold=threshold)
        accuracies.append(accuracy(y_test, y_pred))
        
    best = thresholds[np.argmax(accuracies)]
    print("best threshold", best)

    if plot:
        plt.plot(thresholds, accuracies)
        plt.show()
    return best


if __name__ == "__main__":

    # Hyperparameters 
    dataset = 0 # should be in [0, 1, 2]
    k = 8
    random_state = 42
    Lambda = 1e-5
    model = KernelLogisticRegression(spectrum_kernel, Lambda=Lambda, tol=1e-5, max_iter=100000)

    # lower bound/ upper bound for the threshold search
    lb, ub = 0.0, 1.0 # for Logistic Regression
    # lb, ub = -0.5, 0.5 # for SVM
    #===============================================================================================

    X_seq = pd.read_csv(f"data/train/Xtr{dataset}.csv")
    X = X_seq["seq"].map(lambda s : k_substrings_embedding(s, k=k)).values
    # X = np.genfromtxt(f"data/train/Xtr{dataset}_mat100.csv", delimiter=' ')
    y = pd.read_csv(f"data/train/Ytr{dataset}.csv")["Bound"].values
    y = 2*y - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    ### 

    model.train(X_train, y_train)
    threshold = best_threshold(model, X_test, y_test, lb=lb, ub=ub, precision=20)