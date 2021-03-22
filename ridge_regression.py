import numpy as np



class RidgeRegression:

    def __init__(self, lam):
        self.lam = lam
    
    def fit(self, X, y):
        y = 2 * y - 1
        self.alpha = np.linalg.solve(X.T @ X + self.lam * np.eye(X.shape[1]), X.T @ y)
        return self
    
    def predict(self, X):
        return X @ self.alpha
    
    def predict_class(self, X):
        return ((1 + np.sign(self.predict(X))) / 2).astype(int)
