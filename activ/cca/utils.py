import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def compute_r2(x, y):
    lr = LinearRegression()
    X = x.reshape(-1, 1)
    return r2_score(y, lr.fit(X, y).predict(X))


def compute_cos(x, y):
    return x.dot(y)/np.sqrt((x.dot(x) * y.dot(y)))


def compute_explained_variance(X_cv, Y_cv, Y, perc_var=0.875, method='r2', tol=0.01, comp_method='pca'):
    """
    Args:
        perc_var (float, str):                      the percent variance to keep. if 'auto', pick cutoff
                                                    at where percent variance plateaus
        tol (float):                                the tolerance for calculating plateau
        method (str):                               the method for calculating similarity between PCs and canonical
                                                    variates. options are 'r2' or 'cos'
        comp_method (str):                          the method for calculating the principal components
                                                    options are 'pca' or 'tsvd'


    """
    if method == 'cos':
        sim = lambda x, y: compute_r2(x, y)
    elif method == 'r2':
        sim = lambda x, y: compute_cos(x, y)**2
    else:
        raise ValueError("method must be 'r2' or 'cos'")

    if comp_method == 'pca':
        Y_pca = PCA()
    elif comp_method == 'tsvd':
        Y_pca = TruncatedSVD(n_components=Y.shape[1]-1)
    else:
        raise ValueError("comp_method must be 'pca' or 'tsvd'")

    Y_pca.fit(Y)
    Y_pcs = Y_pca.transform(Y)
    exp_var_ratio = Y_pca.explained_variance_ratio_
    cum_evr = np.cumsum(exp_var_ratio)

    if perc_var == 'auto':
        dx = cum_evr[1:] - cum_evr[:-1]
        n_components = np.where(dx < tol)[0][0]

    else:
        if perc_var > 1:
            perc_var = perc_var/100
        if perc_var < 1.0:
            n_components = np.where(cum_evr >= perc_var)[0][0]
        else:
            n_components = Y_pca.components_.shape[0]

    exp_var_ratio = exp_var_ratio[:n_components]
    Y_pcs = Y_pcs[:, :n_components]

    cv_pc_sim = np.zeros((Y_cv.shape[1], Y_pcs.shape[1]))
    for i in range(Y_cv.shape[1]):
        for j in range(Y_pcs.shape[1]):
            cv_pc_sim[i, j] = sim(Y_cv[:,i], Y_pcs[:, j])

    cv_r2 = np.zeros(Y_cv.shape[1])
    for i in range(Y_cv.shape[1]):
        cv_r2[i] = compute_r2(X_cv[:, i], Y_cv[:, i])

    exp_var = np.sum(cv_pc_sim*exp_var_ratio, axis=1) * cv_r2

    return exp_var, n_components
