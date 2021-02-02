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
                       rx=self.alpha_x, ry=self.alpha_y, return_cov=True,
                       random_state=self.random_state)
        H, S, self.n_iters_ = ret[0], ret[1], ret[2]
        self.x_weights_, self.y_weights_ = H, S
        self.X_cov_ = ret[3]
        self.Y_cov_ = ret[4]
        self.XY_cov_ = ret[5]
        return self

    def transform(self, X, Y):
        if getattr(self, 'X_weights_', None) is None:
            raise ValueError("TALSCCA is not fit")
        return X.dot(self.x_weights_), Y.dot(self.y_weights_)

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


from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.stats import pearsonr

class UoI_sCCA_Base(BaseEstimator):


    def __init__(self, n_components=2, metric='euclidean', cons_meth='median', n_boots=20,
                       subsample_fraction=0.9, cm_kwargs=None, random_state=None, score_function='bic'):
        if cons_meth == 'median':
            self.cons_meth = np.median
            self.cm_kwargs = {'axis': 1}
        elif cons_meth == 'mean':
            self.cons_meth = np.mean
            self.cm_kwargs = {'axis': 1}
        self.random_state = check_random_state(random_state)
        self.n_components = n_components
        self.metric = metric
        self.n_boots = n_boots
        self.subsample_fraction = subsample_fraction
        self.est_cca = CCA(n_components=1, scale=False, )
        if score_function == 'bic':
            self.score_function = self.bic
        elif score_function == 'pearsonr':
            self.score_function = self.pearsonr
        else:
            msg = "Unrecognized score_function: '%s' -- please choose from 'bic' or 'pearsonr'" % score_function
            raise ValueError(msg)

    @classmethod
    def ll(cls, x, y):
        """
        The log likelihood of two canonical variates
        """
        return np.log(sps.norm().pdf(x-y)).sum()

    @classmethod
    def pearsonr(cls, x, y, parameters):
        return sps.pearsonr(x, y)[0]

    @classmethod
    def bic(cls, x, y, parameters):
        k = np.sum(parameters != 0)
        L = cls.ll(x, y)
        n = len(x)
        return k * np.log(n) - 2 * L

    def fit(self, X, Y):
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]
        k = self.n_components
        n_params = len(self.params)

        loading_samples = np.zeros(len(self.params), self.n_boots, k, (p + q))

        idx = np.arange(n)
        boot_samples = np.array([self.random_state.permutation(n) for i in range(self.n_boots)])
        rep_idx = int(self.subsample_fraction * n)

        #################
        # BEGIN selection
        #################
        for param_j in range(n_params):
            for boot_i in range(self.n_boots):
                idx = boot_samples[boot_i, :rep_idx]
                X_rep, Y_rep = X[idx], Y[idx]
                self.sparse_cca.set_params(**self.params[param_j])
                self.sparse_cca.fit(X_rep, Y_rep)
                loading_samples[param_j, boot_i, :, :p] = self.sparse_cca.X_weights_
                loading_samples[param_j, boot_i, :, p:] = self.sparse_cca.Y_weights_

        loading_samples = loading_samples.reshape(len(self.params), self.n_boots*k, p + q)
        models = np.zeros((n_params, k, p + q))
        for param_j in range(n_params):
            # Cluster occupancy vectors
            samples = (loading_samples[param_j] != 0).astype(int)
            labels = cut_tree(linkage(dist(samples, metric=self.metric), method='ward'), n_clusters=k)[:,0]
            for label in np.unique(labels):   # labels <=> components
                models[param_j, label, :] = self.cons_meth(samples[labels==label], **self.cm_kwargs) != 0
        ###############
        # END selection
        ###############

        ##################
        # BEGIN estimation
        ##################
        scores = np.zeros(n_params)
        for model_i in range(n_params):
            for boot_i in range(self.n_boots):
                train_idx = boot_samples[boot_i, :rep_idx]
                test_idx = boot_samples[boot_i, rep_idx:]
                X_weight = np.zeros(k, p)
                Y_weight = np.zeros(k, q)
                X_pred = np.zeros(k*n)
                Y_pred = np.zeros(k*n)
                s = 0
                e = 0
                for component_i in range(k):
                    e += k
                    model = models[model_i, component_i]
                    X_mask, Y_mask = np.array_split(model, [p])
                    X_train = X[train_idx, X_mask]
                    Y_train = Y[train_idx, Y_mask]
                    X_test = X[test_idx, X_mask]
                    Y_test = Y[test_idx, Y_mask]
                    self.est_cca.fit(X_train, Y_train)
                    X_weight[component_i, X_mask] = self.est_cca.x_weights_
                    Y_weight[component_i, Y_mask] = self.est_cca.y_weights_
                    X_pred[s:e], Y_pred[s:e] = self.est_cca.transform(X_test, Y_test)
                    s = e
                estimates[boot_i, model_i, :k*p] = X_weight.flatten()
                estimates[boot_i, model_i, k*p:] = Y_weight.flatten()
                scores[boot_i, model_i] = self.score_function(X_pred, Y_pred, estimates[boot_i, model_i])

        # calculate the best model for each bootstrap
        model_max_idx = np.argmax(scores, axis=1)
        best_estimates = estimates[np.arange(self.n_boots), model_max_idx, :]

        # aggregate across bootstraps
        weights = np.median(best_estimates, axis=0)

        self.x_weights_ = weights[:k*p].reshape(k, p).T
        self.y_weights_ = weights[k*p:].reshape(k, q).T
        ################
        # END estimation
        ################

        return self

    def transform(self, X, Y):
        if getattr(self, 'X_weights_', None) is None:
            raise ValueError("TALSCCA is not fit")
        return X.dot(self.x_weights_), Y.dot(self.y_weights_)
