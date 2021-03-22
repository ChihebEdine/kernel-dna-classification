import numpy as np
from qpsolvers import solve_qp


class KernelRidgeRegression:
    def __init__(self, kernel_func, Lambda=0.1):
        """
        Parameters
        ----------
            kernel_func : lambda function 
                a kernel function, takes two sets of vectors X_1 (n_1, m) and  X_2 (n_2, m)
                and returns the gram matrix K(X_1, X_2) (n_1, n_2)
                
            Lambda : float
                regularization parameter      
        """
        self.Kernel = kernel_func
        self.Alpha = None
        self.Data = None
        self.Lambda = Lambda
        print("Kernel Ridge Regression")
    
    def reset(self):
        self.Alpha = None
        self.Data = None
        
    def train(self, X, y):
        """
        Parameters
        ----------
            X : np.array. shape (n_samples, dim)
                training features
                
            y : np.array. shape (n_samples,)
                training targets
        """
        self.Data = X
        K = self.Kernel(X,X)
        n = K.shape[0]
        I = np.eye(n)
        self.Alpha = np.linalg.solve(K + self.Lambda*n*I, y)
        
    def predict(self, X):
        """
        Parameters
        ----------
            X : np.array. shape (n_samples, dim)
                features
                
        Returns
        -------
            y : np.array. shape (n_samples,)
                prediction
        """
        K = self.Kernel(X, self.Data)
        y = K @ self.Alpha
        return y


class KernelLogisticRegression:
    def __init__(self, kernel_func, Lambda=0.1, tol=1e-5, max_iter=100000):
        """
        Parameters
        ----------
            kernel_func : lambda function 
                a kernel function, takes two sets of vectors X_1 (n_1, m) and  X_2 (n_2, m)
                and returns the gram matrix K(X_1, X_2) (n_1, n_2)
                
            Lambda : float
                regularization parameter
                
            tol : float
                tolerance for stopping criteria
                
            max_iter : int
                maximum number of iterations allowed to converge
        """
        self.Kernel = kernel_func
        self.Alpha = None
        self.Data = None
        self.Lambda = Lambda
        self.tol = tol
        self.max_iter = max_iter
        print("Kernel Logistic Regression")
    
    def reset(self):
        self.Alpha = None
        self.Data = None
        
    def loss(self, s):
        l1 = -1/(1 + np.exp(s))
        l2 = np.exp(s)/(1 + np.exp(s))**2
        return np.diag(l1), np.diag(l2)
        
    def train(self, X, y):
        """
        Parameters
        ----------
            X : np.array. shape (n_samples, dim)
                training features
                
            y : np.array (int). shape (n_samples,)
                training labels in {-1, 1}
        """
        self.Data = X
        K = self.Kernel(X,X)
        n = K.shape[0]
        I = np.eye(n)
        
        alpha_old = np.zeros(n)
        
        for step in range(self.max_iter):
            m = K @ alpha_old
            P, W = self.loss(y*m)
            z = W @ m - P @ y
            alpha_new = np.linalg.solve(W @ K + self.Lambda*n*I, z)
            
            error = np.linalg.norm(alpha_new - alpha_old)
            
            if error < self.tol:
                break
            else:
                alpha_old = alpha_new
                
        if (step == self.max_iter - 1) and (error > self.tol):
            print(f"Kernel Logistic Regression didn't converge ! you might want to take max_iter > {self.max_iter}")
    
        self.Alpha = alpha_new

    def predict(self, X, threshold=0.5):
        """
        Parameters
        ----------
            X : np.array. shape (n_samples, dim)
                features
            
            threshold : float in [0, 1]
                probability threshold to predict 1
                
        Returns
        -------
            y : np.array (int). shape (n_samples, 1)
                predicted labels in {-1, 1}
        """
        K = self.Kernel(X, self.Data)
        f = K @ self.Alpha
        p = 1/(1+np.exp(-f))
        y = (p>threshold).astype(int)
        return 2*y - 1, p


class KernelSVM:
    def __init__(self, kernel_func, Lambda=0.1, reg=1e-10):
        """
        Parameters
        ----------
            kernel_func : lambda function
                a kernel function, takes two sets of vectors X_1 (n_1, m) and  X_2 (n_2, m)
                and returns the gram matrix K(X_1, X_2) (n_1, n_2)

            Lambda : float
                regularization parameter

            reg : float << 1
                small number to add to the diagonal of K
                to ensure that it is positive definite
        """
        self.Kernel = kernel_func
        self.Alpha = None
        self.Data = None
        self.Lambda = Lambda
        self.reg = reg
        print("Kernel Support Vector Machines")
    
    def reset(self):
        self.Alpha = None
        self.Data = None
        
    def train(self, X, y):
        """
        Parameters
        ----------
            X : np.array. shape (n_samples, dim)
                training features
                
            y : np.array (int). shape (n_samples,)
                training labels in {-1, 1}
        """
        self.Data = X
        K = self.Kernel(X,X)
        n = K.shape[0]
        I = np.eye(n)
        K += self.reg*I
        
        bound = 1/(2*self.Lambda*n)
        y = y.astype(float)
        lb =  (bound*y-bound)/2
        ub =  (bound*y+bound)/2
        
        self.Alpha = solve_qp(P=K, q=-y, lb=lb, ub=ub, solver='quadprog')

    def predict(self, X, threshold=0.0):
        """
        Parameters
        ----------
            X : np.array. shape (n_samples, dim)
                features
            
            threshold : float
                probability threshold to predict 1
                
        Returns
        -------
            y : np.array (int). shape (n_samples,)
                predicted labels in {-1, 1}
        """
        K = self.Kernel(X, self.Data)
        f = K @ self.Alpha
        y = (f>threshold).astype(int)
        return 2*y - 1, f
