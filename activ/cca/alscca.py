from sklearn.utils import check_random_state
from sklearn.preprocessing import scale, normalize
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
            t = inprod(Ai, M)
            Ai = Ai - t * Aj
        A[:, i] = Ai / np.sqrt(inprod(Ai, M))
    return A


def cdsolve(X, Y, init, reg, random_state, max_iter=1000, selection='cyclic'):
    ret = np.zeros((X.shape[1], Y.shape[1]), dtype=np.float64)
    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)
    ##init =
    #_, this_coef, this_dual_gap = \
    #    enet_path(X, Y, coef_init=init, alphas=[reg],
    #              random_state=random_state, max_iter=max_iter,
    #              selection=selection,
    #              check_input=False, return_n_iters=False,
    #              l1_ratio=1.0, eps=None, n_alphas=None)

#    precompute = np.empty(shape=(X.shape[1], X.shape[1]), dtype=X.dtype, order='C')
#    np.dot(X.T, X, out=precompute)

    for i in range(Y.shape[1]):
        _, this_coef, this_dual_gap = \
            enet_path(X, Y[:, i], coef_init=init[:, i], alphas=[reg],
                      random_state=random_state, max_iter=max_iter,
                      selection=selection, precompute='auto',
                      check_input=True, return_n_iters=False,
                      l1_ratio=1.0, eps=None, n_alphas=None)
        ret[:, i] = this_coef.squeeze()
    return ret

def tals_cca(X, Y, k, T=100, random_state=None, rx=0.01, ry=0.01):
    random_state = check_random_state(random_state)
    n = len(X)
    p = X.shape[1]
    q = Y.shape[1]
    Cxx = X.T.dot(X)/n + rx*np.identity(p)    # regularize for
    Cyy = Y.T.dot(Y)/n + ry*np.identity(q)    # stability
    Cxy = X.T.dot(Y)/n
    H_t1 = gs(random_state.standard_normal((p, k)), Cxx)
    S_t1 = gs(random_state.standard_normal((q, k)), Cyy)

    H_t = None
    S_t = None

    for t in range(T):
        H_init = H_t1.dot(LA.pinv(inprod(H_t1.T, Cxx)).dot(inprod(H_t1.T, Cxy, S_t1.T)))
        # solve for H_t, initialized at H_init, use S_t1
        H_t = cdsolve(X, Y.dot(S_t1), H_init, rx/2, random_state)
        H_t = gs(H_t, Cxx)
        S_init = S_t1.dot(LA.pinv(inprod(S_t1.T, Cyy)).dot(inprod(S_t1.T, Cxy.T, H_t.T)))
        # solve for S_t, initialized at S_init, use H_t
        S_t = cdsolve(Y, X.dot(H_t), S_init, ry/2, random_state)
        S_t = gs(S_t, Cyy)
        H_t1 = H_t
        S_t1 = S_t

    return H_t, S_t


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
