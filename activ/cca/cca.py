import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LassoCV



class SparseCCA(BaseEstimator):
    def __init__(self, lasso=None, random_state=None, n_components=1, max_iters=100, tol=1e-4):
        self.lasso = LassoCV(n_alphas=10, fit_intercept=False, cv=5, max_iter=20000, tol=0.0001)
        self.random_state = check_random_state(random_state)
        self.n_components = n_components
        self.max_iters = max_iters
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        self.tol = tol

    @staticmethod
    def deflate(A, v):
        return (np.identity(A.shape[0]) - np.outer(v,v)/np.linalg.norm(v)) @ A

    def fit(self, X, Y):

        X_i = self.X_scaler.fit_transform(X)
        Y_i = self.Y_scaler.fit_transform(Y)

        self.X_coef_ = np.zeros((X_i.shape[1], self.n_components))
        self.Y_coef_ = np.zeros((Y_i.shape[1], self.n_components))

        self.X_variates_ = np.zeros((X_i.shape[0], self.n_components))
        self.Y_variates_ = np.zeros((Y_i.shape[0], self.n_components))

        self.n_iters_ = np.zeros(self.n_components)

        self.beta_d_ = np.zeros((self.n_components, self.max_iters))
        self.theta_d_ = np.zeros((self.n_components, self.max_iters))

        for comp in range(self.n_components):
            theta = normalize(self.random_state.rand(Y_i.shape[1],1), norm='l2', axis=0)
            beta = self.lasso.fit(X_i, Y_i@theta).coef_

            i = 0

            for i in range(self.max_iters):
                beta_new = self.lasso.fit(X_i, (Y_i@theta).ravel()).coef_
                theta_new = self.lasso.fit(Y_i, (X_i@beta).ravel()).coef_
                self.beta_d_[comp, i] = np.linalg.norm(beta_new - beta)/beta.shape[0]
                self.theta_d_[comp, i] = np.linalg.norm(theta_new - theta)/theta.shape[0]
                beta = beta_new
                theta = theta_new

                if self.beta_d_[comp, i] <= self.tol and self.theta_d_[comp, i] <= self.tol:
                    break

            self.n_iters_[comp] = i
            self.X_coef_[:, comp] = beta
            self.Y_coef_[:, comp] = theta

            self.X_variates_[:, comp] = X_i @ beta
            self.Y_variates_[:, comp] = Y_i @ theta

            # deflate current X
            X_i = self.deflate(X_i, self.X_variates_[:, comp])
            Y_i = self.deflate(Y_i, self.Y_variates_[:, comp])

        return self

