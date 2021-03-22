import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix



class GaussianRidgeRegression:

    def __init__(self, lam, l):
        self.lam = lam
        self.l = l
    
    def fit(self, X, y):
        y = 2 * y - 1

        self.X_train = X.copy()

        if self.l == 'auto':
            self.l = np.sqrt(0.5 * X.shape[1] * X.var())
        
        n = X.shape[0]
        K = np.exp(- 0.5 * squareform(pdist(X) ** 2)) / (self.l ** 2)

        self.alpha = np.linalg.solve(K + self.lam * n * np.eye(n), y)

        return self
    
    def predict(self, X):
        kx = np.exp(-0.5 * (distance_matrix(X, self.X_train) ** 2)) / (self.l ** 2)
        return kx @ self.alpha
    
    def predict_class(self, X):
        return ((1 + np.sign(self.predict(X))) / 2).astype(int)

