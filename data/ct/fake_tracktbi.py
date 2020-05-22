import pandas as pd
import numpy as np

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('tracktbi_csv', help='original TRACK-TBI csv file')

args = parser.parse_args()

np.random.seed(15)

orig_df = pd.read_csv(args.tracktbi_csv)


df = orig_df.copy()

for colname in orig_df.columns:
    if colname == 'PatientNum':
        continue
    col = orig_df[colname].values
    idx = np.random.randint(len(col), size=len(col))
    df[colname] = col[idx]

df.to_csv(sys.stdout, index=False, sep=',')
