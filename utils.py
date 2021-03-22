import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from models import KernelRidgeRegression, KernelLogisticRegression, KernelSVM
from kernels import linear_kernel, gaussian_kernel


def accuracy(y_test, y_pred):
    return (y_pred == y_test).astype(int).mean()


def best_threshold(model, X_test, y_test, lb=-0.5, ub=0.5, precision=20, plot=True):
    thresholds = np.linspace(lb, ub, precision)
    accuracies = []
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
    i = 2

    #X_seq = pd.read_csv(f"data/train/Xtr{i}.csv")
    X = np.genfromtxt(f"data/train/Xtr{i}_mat100.csv", delimiter=' ')
    y = pd.read_csv(f"data/train/Ytr{i}.csv")["Bound"].values
    y = 2*y - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KernelLogisticRegression(gaussian_kernel, Lambda=1e-5, tol=1e-5, max_iter=100000)
    model.train(X_train, y_train)
    threshold = best_threshold(model, X_test, y_test, lb=-0.5, ub=0.5, precision=20)
