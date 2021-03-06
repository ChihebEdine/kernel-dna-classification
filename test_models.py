import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import KernelRidgeRegression, KernelLogisticRegression, KernelSVM
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
    y = y.reshape(-1)

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
    y = y.reshape(-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    KLR.train(X_train, y_train)
    y_pred, p = KLR.predict(X_test)
    plt.scatter(X_test, y_test, label="ground truth")
    plt.scatter(X_test, y_pred, label="prediction", marker='.')
    plt.scatter(X_test, p, label="probabilies P(1|X)", marker='.')
    plt.title(f"Kernel Logistic Regression ({k})")
    plt.legend()
    plt.show()


def test_KernelSVM(k="gaussian"):
    if k == "gaussian":
        SVM = KernelSVM(gaussian_kernel, Lambda=0.01)
    elif k == "linear":
        SVM = KernelSVM(linear_kernel, Lambda=0.01)
    else:
        return
    
    X = np.random.uniform(low=-1.0, high=1.0, size=(100,2))
    y = 2*(X[:,0]**2 + X[:,1]**2<0.5).astype(int) - 1

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    SVM.train(X_train, y_train)

    color = ["blue"  if i==1 else "red" for i in y_test]
    ax1.scatter(X_test[:,0], X_test[:,1], c=color)
    ax1.set_title(f"Ground truth coloring")
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])

    y_pred, _ = SVM.predict(X_test)
    c = ["blue"  if i==1 else "red" for i in y_pred ]
    ax2.scatter(X_test[:,0], X_test[:,1], c=c)
    ax2.set_title(f"Kernel Support Vector Machines\n({k})")
    ax2.set_xlim([-1.1, 1.1])
    ax2.set_ylim([-1.1, 1.1])

    plt.show()



if __name__ == "__main__":
    test_KernelRidgeRegression(k="gaussian")
    test_KernelRidgeRegression(k="linear")
    test_KernelLogisticRegression(k="gaussian")
    test_KernelLogisticRegression(k="linear")
    test_KernelSVM(k="gaussian")
    test_KernelSVM(k="linear")