import numpy as np
import pandas as pd

def load_data(tbi_measures_path, orig_tbi_data_path):

    sdf = pd.read_csv(orig_tbi_data_path, header=0, index_col=0)
    df = pd.read_csv(tbi_measures_path, index_col=0)

    ptype = df['PatientType']
    categories = np.unique(ptype)
    measures = df.drop('PatientType', axis=1).columns

    df_patients = set(df.index)
    sdf_patients = set(sdf.index)
    bad_patients = list(df_patients ^ sdf_patients)
    df = df.drop(df_patients - sdf_patients, axis=0)
    sdf = sdf.drop(sdf_patients - df_patients, axis=0)

    df['GCS'] = sdf['GCSMildModSevereRecode']

    df['GCS'][np.where(df['GCS'] == ' ')[0]] = 'unmeasurable'
    return df


if __name__ == '__main__':
    import argparse

    epi = """
    """
    parser = argparse.ArgumentParser(description='Parse data from CT registration', epilog=epi,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ct_measures_csv', type=str, help='CSV of CT measures')
    parser.add_argument('tracktbi_csv', type=str, help='CSV of original TRACK-TBI data')
    parser.add_argument('--original', action='store_true', default=False, help='input_dir is directory containing original inputs')

    args = parser.parse_args()
    df = load_data(args.ct_measures_csv, args.tracktbi_csv)

