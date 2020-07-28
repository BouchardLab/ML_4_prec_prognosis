import argparse
import pandas as pd

from activ.ct.summarize import load_data
from activ.readfile import merge_data

parser = argparse.ArgumentParser()
parser.add_argument('cleaned', help='path to cleaned TRACK-TBI data, in HDF5 format')
parser.add_argument('output', help='path to save data to')
parser.add_argument('-c', '--ct_path', help='path to preprocessed CT data, in CSV format')
parser.add_argument('-r', '--scalar_path', help='path to original TRACK-TBI data, in CSV format')
parser.add_argument('-t', '--connectome_path', help='path to original connectome data, in CSV format')
parser.add_argument('-d', '--data_dict_path', help='path to the data dictionary file')

args = parser.parse_args()

ct_df = None
if args.ct_path:
    ct_df, ctlabels = load_data(args.ct_path, args.scalar_path, filter_suffixes=('Sum', 'Min', 'Max', 'Q05', 'Q95', 'GEO'))

cntm_df = None
if args.connectome_path:
    cntm_df = pd.read_csv(args.connectome_path, header=0, index_col=0)

merge_data(args.cleaned, args.output, ct_df=ct_df, cntm_df=cntm_df, data_dict_path=args.data_dict_path)
