from sklearn.utils import check_random_state
from sklearn.preprocessing import scale, normalize
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator
import numpy as np


class ALSCCA(BaseEstimator):
    """
    Alternating Least Squares Canonical Correlation Analysis
    """


    def __init__(self, X_lm=None, Y_lm = None, random_state=None, n_components=1, max_iters=100, alpha=0.1):
        """
        Args:
            X_lm:                     Linear model for regressing X onto Y
            Y_lm:                     Linear model for regressing Y onto X
            max_iter:                 max_iter for default Lasso object
            alpha:                    alpha for default Lasso object
        """
        if X_lm is None:
            X_lm = Lasso(alpha=alpha, max_iter=2000)
        if Y_lm is None:
            Y_lm = Lasso(alpha=alpha, max_iter=2000)

        self.X_lm = X_lm
        self.Y_lm = Y_lm

        self.random_state = check_random_state(random_state)
        self.n_components = n_components
        self.max_iters = max_iters

    def fit(self, X, Y):

        theta = normalize(self.random_state.rand(Y.shape[1],1), norm='l2', axis=0)

        beta = self.X_lm.fit(X, Y@theta).coef_

        n_iters = 10
        beta_d = np.zeros(n_iters)
        theta_d = np.zeros(n_iters)

        for i in range(n_iters):
            beta_new = self.X_lm.fit(X, Y@theta).coef_
            theta_new = self.Y_lm.fit(Y, X@beta).coef_
            beta_d[i] = np.linalg.norm(beta_new - beta)
            theta_d[i] = np.linalg.norm(theta_new - theta)
            beta = beta_new
            theta = theta_new
        self.beta = beta
        self.theta = theta
        return self

    def transform(self, X, Y):
        z = X @ self.beta
        s = Y @ self.theta
        return z, s
