import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kernels import linear_kernel, gaussian_kernel
from models import KernelRidgeRegression, KernelLogisticRegression, KernelSVM
from utils import accuracy


def train_and_predict(model, threshold, dataset=0):
    #X_seq = pd.read_csv(f"data/train/Xtr{dataset}.csv")
    X = np.genfromtxt(f"data/train/Xtr{dataset}_mat100.csv", delimiter=' ')
    y = pd.read_csv(f"data/train/Ytr{dataset}.csv")["Bound"].values
    y = 2*y - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.train(X_train, y_train)
    y_pred, _ = model.predict(X_test, threshold=threshold)
    print("Test accuracy = ", accuracy(y_test, y_pred))
    
    X_seq_te = pd.read_csv(f"data/test/Xte{dataset}.csv")
    X_te = np.genfromtxt(f"data/test/Xte{dataset}_mat100.csv", delimiter=' ')
    y_pred_te, _ = model.predict(X_te, threshold=threshold)
    
    X_seq_te["Bound"] = (y_pred_te+1)//2
    return X_seq_te


if __name__ == "__main__":
    model0 = KernelSVM(gaussian_kernel, Lambda=1e-5, reg=1e-10)
    threshold0 = -0.23

    model1 = KernelSVM(gaussian_kernel, Lambda=1e-5, reg=1e-10)
    threshold1 = -0.45

    model2 = KernelLogisticRegression(gaussian_kernel, Lambda=1e-5, tol=1e-5, max_iter=100000)
    threshold2 = 0.44

    X_seq_te0 = train_and_predict(model=model0, threshold=threshold0, dataset=0)
    X_seq_te1 = train_and_predict(model=model1, threshold=threshold1, dataset=1)
    X_seq_te2 = train_and_predict(model=model2, threshold=threshold2, dataset=2)

    FINAL = pd.concat([X_seq_te0, X_seq_te1, X_seq_te2], ignore_index=True)[["Id", "Bound"]]
    FINAL.to_csv("data/submission.csv", index=False)
