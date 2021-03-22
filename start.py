import pandas as pd
from sklearn.model_selection import train_test_split
from kernels import spectrum_kernel, k_substrings_embedding
from models import KernelSVM
from utils import accuracy


def train_and_predict_advanced(model, threshold, k, dataset=0):
    
    X_seq = pd.read_csv(f"data/train/Xtr{dataset}.csv")
    X = X_seq["seq"].map(lambda s : k_substrings_embedding(s, k=k)).values
    y = pd.read_csv(f"data/train/Ytr{dataset}.csv")["Bound"].values
    y = 2*y - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.train(X_train, y_train)
    
    y_pred, _ = model.predict(X_test, threshold=threshold)
    print(f"[Data set {dataset}] Test accuracy = ", accuracy(y_test, y_pred))
    
    X_seq_te = pd.read_csv(f"data/test/Xte{dataset}.csv")
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

    X_seq_te0 = train_and_predict_advanced(model=model0, threshold=threshold0, k=k0, dataset=0)
    X_seq_te1 = train_and_predict_advanced(model=model1, threshold=threshold1, k=k1, dataset=1)
    X_seq_te2 = train_and_predict_advanced(model=model2, threshold=threshold2, k=k2, dataset=2)

    FINAL = pd.concat([X_seq_te0, X_seq_te1, X_seq_te2], ignore_index=True)[["Id", "Bound"]]
    FINAL.to_csv("data/final_submission.csv", index=False)