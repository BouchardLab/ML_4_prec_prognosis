from sklearn.utils import check_random_state
from sklearn.preprocessing import normalize
from sklearn.linear_model import Lasso
from sklearn.base import BaseEstimator
import numpy as np
import numpy.linalg as LA
from sklearn.linear_model import enet_path


def inprod(u, M, v=None):
    v = u if v is None else v
    return u.dot(M.dot(v.T))


def gs(A, M):
    """
    Gram-Schmidt with inner product for M
    """
    A = A.copy()
    A[:, 0] = A[:, 0] / np.sqrt(inprod(A[:, 0], M))
    for i in range(1, A.shape[1]):
        Ai = A[:, i]
        for j in range(0, i):
            Aj = A[:, j]
            t = inprod(Ai, M, Aj)
            Ai = Ai - t * Aj
        A[:, i] = Ai / np.sqrt(inprod(Ai, M))
    return A


def cdsolve(X, Y, init, reg, random_state, max_iter=1000, tol=0.1, selection='cyclic'):
    ret = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float64)
    for i in range(Y.shape[1]):
        _, this_coef, this_dual_gap = \
            enet_path(X, Y[:, i], coef_init=init[:, i], alphas=[reg],
                      random_state=random_state, max_iter=max_iter,
                      selection=selection, precompute='auto',
                      check_input=True, return_n_iters=False,
                      tol=0.1,
                      l1_ratio=1.0, eps=None, n_alphas=None)
        ret[:, i] = this_coef.squeeze()
    return ret

def tals_cca(X, Y, k, max_iters=1000, tol=0.0001, random_state=None, rx=0.001, ry=0.001,
             return_cov=False):
    random_state = check_random_state(random_state)
    n = len(X)
    p = X.shape[1]
    q = Y.shape[1]
    Cxx = X.T.dot(X)/n + rx*np.identity(p)    # regularize for
    Cyy = Y.T.dot(Y)/n + ry*np.identity(q)    # stability
    Cxy = X.T.dot(Y)/n
    H_0 = random_state.standard_normal((p, k))
    S_0 = random_state.standard_normal((q, k))
    H_t1 = gs(H_0, Cxx)
    S_t1 = gs(S_0, Cyy)

    H_t = None
    S_t = None

    n_iters = 0
    for _ in range(max_iters):
        H_init = H_t1.dot(LA.pinv(inprod(H_t1.T, Cxx)).dot(inprod(H_t1.T, Cxy, S_t1.T)))
        # solve for H_t, initialized at H_init, use S_t1
        H_t = cdsolve(X, Y.dot(S_t1), H_init, rx/2, random_state)
        H_t = gs(H_t, Cxx)
        S_init = S_t1.dot(LA.pinv(inprod(S_t1.T, Cyy)).dot(inprod(S_t1.T, Cxy.T, H_t.T)))
        # solve for S_t, initialized at S_init, use H_t
        S_t = cdsolve(Y, X.dot(H_t), S_init, ry/2, random_state)
        S_t = gs(S_t, Cyy)
        n_iters += 1
        delta = LA.norm(H_t - H_t1) + LA.norm(S_t - S_t1)
        if delta < tol:
            break
        H_t1 = H_t
        S_t1 = S_t

    ret = [H_t, S_t, n_iters]
    if return_cov:
        ret.append(Cxx)
        ret.append(Cyy)
        ret.append(Cxy)
    return tuple(ret)



class TALSCCA(BaseEstimator):
    """
    Alternating Least Squares Canonical Correlation Analysis
    """

    def __init__(self, random_state=None,
                 n_components=1, max_iters=100, alpha_x=0.001,
                 alpha_y=0.001):
        """
        Args:
            X_lm:                     Linear model for regressing X onto Y
            Y_lm:                     Linear model for regressing Y onto X
            max_iter:                 max_iter for default Lasso object
            alpha:                    alpha for default Lasso object
        """

        self.random_state = check_random_state(random_state)
        self.n_components = n_components
        self.max_iters = max_iters
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

    def fit(self, X, Y):
        ret = tals_cca(X, Y, self.n_components, max_iters=self.max_iters,
                       rx=self.alpha_x, ry=self.alpha_y, return_cov=True)
        H, S, self.n_iters_ = ret[0], ret[1], ret[2]
        self.X_components_, self.Y_components_ = H.T, S.T
        self.X_cov_ = ret[3]
        self.Y_cov_ = ret[4]
        self.XY_cov_ = ret[5]
        return self

    def transform(self, X, Y):
        if getattr(self, 'X_components_', None) is None:
            raise ValueError("TALSCCA is not fit")
        return X.dot(self.X_components_.T), Y.dot(self.Y_components_.T)

    def fit_transform(self, X, Y):
        return self.fit(X,Y).transform(X,Y)

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

        theta = normalize(self.random_state.rand(Y.shape[1], self.n_components), norm='l2', axis=0)

        beta = self.X_lm.fit(X, Y@theta).coef_.T

        n_iters = 10
        beta_d = np.zeros(n_iters)
        theta_d = np.zeros(n_iters)

        for i in range(n_iters):
            theta_new = self.Y_lm.fit(Y, X@beta).coef_.T
            beta_new = self.X_lm.fit(X, Y@theta).coef_.T
            beta_d[i] = np.linalg.norm(beta_new - beta)
            theta_d[i] = np.linalg.norm(theta_new - theta)
            beta = beta_new
            theta = theta_new
        self.beta = beta
        self.theta = theta
        self.beta_d = beta_d
        self.theta_d = theta_d
        return self

    def transform(self, X, Y):
        z = X @ self.beta
        s = Y @ self.theta
        return z, s
