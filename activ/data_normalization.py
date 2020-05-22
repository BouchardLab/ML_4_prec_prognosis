import sklearn.preprocessing as _skp

def data_normalization(M, method):
    if method == 'z-score' or method == 'standard':
        ret = _skp.scale(M)
    elif method == 'positive':
        ret = _skp.minmax_scale(M)
    else:
        raise ValueError("method must be 'z-score'/'standard' or 'positive'")
    return ret
