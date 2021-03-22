import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix


from numba import jit


class GaussianLogisticRegression:
    def __init__(self, lam, l):
        self.lam = lam
        self.l = l
        
        self.loss = lambda u: - np.log(expit(u))
        self.dloss = lambda u: - expit(- u)
        self.d2loss = lambda u: expit(u) * expit(- u)
    
    
    def fit(self, X, y):
        y = 2 * y - 1

        self.X_train = X.copy()
        
        if self.l == 'auto':
            self.l = np.sqrt(0.5 * X.shape[1] * X.var())
            
        n = X.shape[0]
        K = np.exp(-0.5 * squareform(pdist(X) ** 2) / (self.l ** 2))
        self.alpha = _klr_fit_raw(np.asfortranarray(K), y.astype(np.float64), self.lam, self.l)

        return self
    
    def predict(self, X):
        kx = np.exp(-0.5 * (distance_matrix(X, self.X_train) ** 2) / (self.l ** 2))
        return kx @ self.alpha
        
    def predict_class(self, X): 
        return ((1 + np.sign(self.predict(X))) / 2).astype(int)
    


@jit(nopython=True, cache=True)
def expit(x):
    return 1 / (1 + np.exp(- x))
    
@jit(nopython=True, cache=True)
def _klr_fit_raw(K, y, lam, l):
    n = K.shape[0]
                          
    alpha = np.asfortranarray(np.zeros((n, 1)))
    
    yka = y * (K @ alpha)
    w = expit(yka) * expit(- yka)
    p = - expit(- yka)
    
    z = K @ alpha - np.linalg.solve(np.diag(w.reshape(-1)), np.diag(p.reshape(-1)) @ y)

    while True:
        # weighted kernel ridge regression
        W_sq = np.diag(np.sqrt(w.reshape(-1)))
        W_inv_sq = np.diag(1 / np.sqrt(w.reshape(-1)))
        new_alpha = np.linalg.solve((W_sq @ K @ W_sq + lam * n * np.eye(n)) @ W_inv_sq, W_sq @ z)

        if np.abs(alpha - new_alpha).max() < 1e-6:
            return alpha

        alpha = new_alpha

        m = K @ alpha
        ym = y * m
        p = - expit(- ym)
        w = expit(ym) * expit(- ym)
        z = m - p * y / w
        