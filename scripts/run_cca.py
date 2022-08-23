import argparse
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from activ import TrackTBIFile
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.preprocessing import scale
from scipy.stats import pearsonr
import warnings
from activ.cca.alscca import TALSCCA


parser = argparse.ArgumentParser()
parser.add_argument('tbifile', type=str, help='the TrackTBIFile')
parser.add_argument('output', type=str, help='the output HDF5 file')
parser.add_argument('-I', '--max_iter', type=int, help='the max number of iterations', default=1000)

args = parser.parse_args()

path = args.tbifile
tbifile = TrackTBIFile(path)
gcs_simple = tbifile.biomarkers[:, np.where(tbifile.biomarker_features == 'GCSMildModSevereRecode')[0][0]]

X = scale(tbifile.biomarkers, with_std=False)
Y = scale(tbifile.outcomes, with_std=False)

print("Running TALSCCA", file=sys.stderr)
params = {'l1_ratio_y': 1.0, 'l1_ratio_x': 1.0, 'alpha_y': 0.5, 'alpha_x': 0.5, 'n_components': 177, 'max_iter': args.max_iter}
talscca = TALSCCA(**params, random_state=1001, verbose=True)
with warnings.catch_warnings(record=False):
    warnings.simplefilter('ignore')
    cv_bm, cv_oc = talscca.fit_transform(X, Y)

print(f"Took {talscca.n_iter_} iterations")
