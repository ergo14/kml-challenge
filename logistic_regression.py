import numpy as np
from scipy.special import expit


class LogisticRegression:

    def __init__(self, beta=0.9, eps=1e-10, max_iterations=200, noisy=False):
        self.beta = beta
        self.eps = eps
        self.max_iterations = max_iterations
        self.noisy = noisy

    def fit(self, X, y):
        self.theta, iterations = newton_iteration(X, y, beta=self.beta, eps=self.eps, max_iterations=self.max_iterations, noisy=self.noisy)
        if self.noisy:
            print('Reporting final number of iterations:', iterations)
        return self
    
    def predict(self, X):
        return X @ self.theta
    
    def predict_class(self, X):
        return ((1 + np.sign(self.predict(X))) / 2).astype(int)



def newton_iteration(X, y, beta, eps, max_iterations, noisy):
    theta = np.zeros((X.shape[1], 1))

    if noisy:
        print('NLL at iteration #1:', compute_nll(theta, X, y))

    iteration = 1
    dt = newton_step(theta, X, y)

    while np.linalg.norm(dt) > eps and iteration < max_iterations:
        rho = 1
        nll = compute_nll(theta + dt, X, y)
        new_nll = compute_nll(theta + rho * dt, X, y)
        while new_nll < nll and rho > 1e-10:
            rho *= beta
            nll = new_nll
            new_nll = compute_nll(theta + rho * beta * dt, X, y)
        theta += rho * dt
        if noisy:
            print(f'NLL at iteration #{iteration}:', compute_nll(theta, X, y))
        dt = newton_step(theta, X, y)
        iteration += 1
    
    return theta, iteration


def newton_step(theta, X, y):
    p = expit(X @ theta)
    grad = - X.T @ (y - p)
    hess = X.T @ np.diag((p * (1 - p))[:, 0]) @ X
    return np.linalg.lstsq(- hess, grad, rcond=None)[0]


def compute_nll(theta, X, y):
    """
    Negative Log Likelihood.
    """
    log_exp = - np.log(expit(- X @ theta))
    if np.any(np.isnan(log_exp) | np.isinf(log_exp)):
        log_exp = X @ theta 
    return - np.sum(y * X @ theta - log_exp)




