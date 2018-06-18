import numpy as np

def data_normalization(M, method):
	if method == 'z-score':
		data_matrix_normalized = (M-mean(M))/np.std(M)
	else if method == 'positive':
		norm_M = M-min(M)
		data_matrix_normalized = norm_M/max(norm_M)
	else:
		print('...................................\n')
        print('error: invalid parameter - method\n')
        print("please choose 'z-score' or 'positive'\n")
        print('...................................\n')

#data_matrix_normalized = (M-repmat(mean(M,1),size(M,1),1))./repmat(std(M,0,1),size(M,1),1);
# M: (row,col): (samples, features)
# mean(M,1): mean of columns
#size(M,1): row of M


#data_matrix_normalized = (M-repmat(min(M,[],1),size(M,1),1))./(repmat(max(M,[],1),size(M,1),1)-repmat(min(M,[],1),size(M,1),1));
