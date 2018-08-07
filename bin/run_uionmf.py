from PyUoI.UoINMF import UoINMF
from activ import load_data
from activ.data_normalization import data_normalization
from sklearn.cluster import DBSCAN
from sklearn.decomposition import NMF
from hdbscan import HDBSCAN
from datetime import  datetime

import h5py
import sys
import os
import argparse

def log(msg):
    print("%s - %s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg), file=sys.stderr)

parser = argparse.ArgumentParser(usage="%(prog)s [options]")
parser.add_argument('-b', '--bootstraps', type=int, help='the number of bootstraps to run', default=50)
parser.add_argument('-m', '--min_cluster_size', type=int, help='the minimum number samples for a cluster', default=20)
parser.add_argument('-o', '--output', type=str, help='the HDF5 file to save results to', default='uoinmf.h5')
parser.add_argument('-f', '--force', action='store_true', help='force overwrite of output file', default=False)

args = parser.parse_args()

if os.path.exists(args.output):
    if not args.force:
        sys.stderr.write("%s exists... cautiously exiting\n" % args.output)
        sys.exit(2)
    else:
        os.remove(args.output)

output = args.output

start = datetime.now()
log('loading data')
data = load_data()
log('normalizing data')
oc = data_normalization(data.outcomes, 'positive')
uoinmf_big = UoINMF(n_bootstraps_i=50, ranks=list(range(2,20)), dbscan=HDBSCAN(min_cluster_size=20, core_dist_n_jobs=1))
log('running UoINMF')
log(repr(uoinmf_big))
uoinmf_big.fit(oc)

log('writing results to %s' % output)
f = h5py.File(output, 'w')
f.create_dataset('outcome_bases', data=uoinmf_big.components_)
f.create_dataset('outcome_bases_samples', data=uoinmf_big.bases_samples_)
f.close()
end = datetime.now()
log('done - found %d bases' % uoinmf_big.components_.shape[0])
log('time elapsed: %s' % str(end-start))
