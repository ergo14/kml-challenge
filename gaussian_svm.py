import numpy as np
import cvxpy as cp

from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix



class GaussianSVM:
    
    def __init__(self, C, l):
        self.C = C
        self.l = l
    
    def fit(self, X, y):
        y = 2 * y - 1

        if self.l == 'auto':
            self.l = np.sqrt(0.5 * X.shape[1] * X.var())

        self.X_train = X.copy()
        n = X.shape[0]
        
        K = np.exp(-0.5 * squareform(pdist(X) ** 2) / (self.l ** 2))        
        Y = np.diag(y.squeeze())

        constraint_matrix = np.r_[Y, -Y]
        constraint_vector = np.concatenate([self.C * np.ones(n), np.zeros(n)])
        
        alpha = cp.Variable(n)
        dual = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(alpha, K) - y.T @ alpha),
            [constraint_matrix @ alpha <= constraint_vector],
        )
        dual.solve()
        
        self.alpha = alpha.value.reshape(-1, 1)
        
        return self
    
    def predict(self, X):
        kx = np.exp(-0.5 * (distance_matrix(X, self.X_train) ** 2) / (self.l ** 2))
        return kx @ self.alpha

    def predict_class(self, X):
        return ((1 + np.sign(self.predict(X))) / 2).astype(int)
        


