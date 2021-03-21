import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import KernelRidgeRegression, KernelLogisticRegression
from kernels import linear_kernel, gaussian_kernel


def test_KernelRidgeRegression(k="gaussian"):

    if k == "gaussian":
        KRR = KernelRidgeRegression(gaussian_kernel, Lambda=0.01)
    elif k == "linear":
        KRR = KernelRidgeRegression(linear_kernel, Lambda=0.01)
    else:
        return

    X = np.linspace(-np.pi, np.pi, 1000).reshape(1000, 1)
    y = 3 * np.sin(X) + np.random.normal(0, 0.3, size=(1000,1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    KRR.train(X_train, y_train)
    y_pred = KRR.predict(X_test)
    plt.scatter(X_test, y_test, label="ground truth")
    plt.scatter(X_test, y_pred, label="prediction")
    plt.title(f"Kernel Ridge Regression ({k})")
    plt.legend()
    plt.show()


def test_KernelLogisticRegression(k="gaussian"):
    
    if k == "gaussian":
        KLR = KernelLogisticRegression(gaussian_kernel, Lambda=0.01)
    elif k == "linear":
        KLR = KernelLogisticRegression(linear_kernel, Lambda=0.01)
    else:
        return

    X = np.linspace(-np.pi, np.pi, 1000).reshape(1000, 1)
    y = 2*(X>0).astype(int) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    KLR.train(X_train, y_train)
    y_pred = KLR.predict(X_test)
    plt.scatter(X_test, y_test, label="ground truth")
    plt.scatter(X_test, y_pred, label="prediction")
    plt.title(f"Kernel Logistic Regression ({k})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_KernelRidgeRegression(k="gaussian")
    test_KernelRidgeRegression(k="linear")
    test_KernelLogisticRegression(k="gaussian")
    test_KernelLogisticRegression(k="linear")