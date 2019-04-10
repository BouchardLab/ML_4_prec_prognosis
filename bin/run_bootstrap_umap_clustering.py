from activ import load_data
import math
from mpi4py import MPI

from activ.clustering import bootstrapped_umap_clustering


data = load_data()
bm = data.biomarkers
oc = data.outcomes
nboot = 50
cluster_sizes = list(range(2, 11))

comm = MPI.COMM
rank = comm.Get_rank()
size = comm.Get_size()

#iterations = range(rank, n_bootstraps, size)

portion = math.ceil(n_bootstraps - rank / size)

print(rank, portion)

#labels, predictions, rand_labels, rand_predictions = bootstrapped_umap_clustering(bm, oc, nboot, cluster_sizes, n_umap_iters=5)


