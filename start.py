import pandas as pd
from kernels import spectrum_kernel, k_substrings_embedding
from models import KernelSVM
import numpy as np


def train_and_predict(model, threshold, k=None, dataset=0):

    if k is None:
        X = np.genfromtxt(f"data/train/Xtr{dataset}_mat100.csv", delimiter=' ')
    else:
        X_seq = pd.read_csv(f"data/train/Xtr{dataset}.csv")
        X = X_seq["seq"].map(lambda s : k_substrings_embedding(s, k=k)).values

    y = pd.read_csv(f"data/train/Ytr{dataset}.csv")["Bound"].values
    y = 2*y - 1
    
    model.train(X, y)

    X_seq_te = pd.read_csv(f"data/test/Xte{dataset}.csv")
    if k is None:
        X_te = np.genfromtxt(f"data/test/Xte{dataset}_mat100.csv", delimiter=' ')
    else:
        X_te = X_seq_te["seq"].map(lambda s : k_substrings_embedding(s, k=k)).values

    y_pred_te, _ = model.predict(X_te, threshold=threshold)
    
    X_seq_te["Bound"] = (y_pred_te+1)//2
    return X_seq_te



if __name__ == "__main__":

    model0 = KernelSVM(spectrum_kernel, Lambda=1e-5, reg=1e-10)
    threshold0 = -0.08
    k0 = 12

    model1 = KernelSVM(spectrum_kernel, Lambda=1e-5, reg=1e-10)
    threshold1 = -0.08
    k1 = 8

    model2 = KernelSVM(spectrum_kernel, Lambda=1e-5, reg=1e-10)
    threshold2 = 0.14
    k2 = 9

    print("Training on the first data set ...")
    X_seq_te0 = train_and_predict(model=model0, threshold=threshold0, k=k0, dataset=0)

    print("Training on the second data set ...")
    X_seq_te1 = train_and_predict(model=model1, threshold=threshold1, k=k1, dataset=1)

    print("Training on the third data set ...")
    X_seq_te2 = train_and_predict(model=model2, threshold=threshold2, k=k2, dataset=2)

    FINAL = pd.concat([X_seq_te0, X_seq_te1, X_seq_te2], ignore_index=True)[["Id", "Bound"]]
    FINAL.to_csv("data/final_submission.csv", index=False)
    print("Done ! check the submission file in the folder data/")