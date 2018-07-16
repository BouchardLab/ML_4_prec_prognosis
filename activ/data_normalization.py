import numpy as np

def data_normalization(M, method):
    ret = None
    if method == 'z-score':
        ret = (M-np.mean(M))/np.std(M)
    elif method == 'positive':
        ret = np.copy(M)
        mn = np.amin(ret, axis=0)
        for i in range(ret.shape[0]):
            ret[i] -= mn
        mx = np.amax(ret, axis=0)
        for i in range(ret.shape[0]):
            ret[i] /= mx
    else:
        raise ValueError("method must be 'z-score' or 'positive'")
        # print('...................................\n')
        # print('error: invalid parameter - method\n')
        # print("please choose 'z-score' or 'positive'\n")
        # print('...................................\n')
    return ret

#data_matrix_normalized = (M-repmat(mean(M,1),size(M,1),1))./repmat(std(M,0,1),size(M,1),1);
# M: (row,col): (samples, features)
# mean(M,1): mean of columns
#size(M,1): row of M


#data_matrix_normalized = (M-repmat(min(M,[],1),size(M,1),1))./(repmat(max(M,[],1),size(M,1),1)-repmat(min(M,[],1),size(M,1),1));
