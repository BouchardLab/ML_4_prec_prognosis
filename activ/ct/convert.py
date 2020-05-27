import pandas as pd
import glob
import os


def read_file(glob_path, prefix, cols):
    """
    Args:
        glob_path: the glob string of the file to read in
        prefix:  the prefix to add to the column names
        cols: the column names to use. Note: the first len(cols) will be read

    Returns:
        A pd.DataFrame with the data aggregated from all files in the glob
    """
    patient_data = list()
    patient_ids = list()

    for path in glob.glob(glob_path):
        ptype = os.path.basename(os.path.dirname(path))
        feats = None
        with open(path, 'r') as f:
            for l_i, l in enumerate(f.readlines()):
                if l.strip().startswith('Label'):
                    prev_idx = 0
                    if feats is not None:
                        patient_data.append(feats)
                    feats = {"PatientType": ptype}
                elif len(l) == 1:
                    patient_data.append(feats)
                    feats = {"PatientType": ptype}
                elif 'Proc' in l:
                    if len(feats) > 1:
                        patient_data.append(feats)
                        feats = {}
                    #i = l.rfind('-')
                    #l[l.rfind(' ')+1:-1]
                    patient_ids.append(l[l.rfind(' ')+1:-1])
                else:
                    data = l.split()
                    label = int(data[0])
                    if label != prev_idx + 1:
                        raise ValueError(f'file {path}, line {l_i}')
                    prev_idx = label
                    data = data[1:]
                    for i, col in enumerate(cols):
                        feat_name = f"{prefix}_{col}_{label}"
                        if data[i] == 'file':
                            breakpoint()
                        feats[feat_name] = float(data[i])
    if len(patient_data) == 0:
        feats.pop('PatientType')
        patient_data.append(feats)
        patient_ids = ['Measure']
    ret = pd.DataFrame(data=patient_data, index=patient_ids)

    return ret


def read_all(*files):
    """
    Args:
        files: a list of tuples containing arguments to *read_file*

    Returns:
        An aggregated pd.DataFrame with data from all input files
    """
    dfs = list()
    for glob, pfx, cols in files:
        dfs.append(read_file(glob, pfx, cols))

    ptypes = dfs[0]['PatientType']
    dfs = [df.drop('PatientType', axis=1) for df in dfs]
    dfs.append(ptypes)

    final_df = pd.concat(dfs,  axis=1)
    return final_df


def main(argv):
    import argparse

    epi = """
input_dir is expected to have directories that correspond to patient type. Within each
of these subdirectories, data for geometric measures and image measures is expected.

If --group6 and --sep_atlas are used, the following files are expected:

    outputLabelGeometryMeasures_21Label_AFF.txt   (AFF_S)
    outputLabelGeometryMeasures_48Label_AFF.txt   (AFF_C)
    outputLabelGeometryMeasures_21Label_PHY.txt   (PHY_S)
    outputLabelGeometryMeasures_48Label_PHY.txt   (PHY_C)
    outputImageIntensityStat_21Atlas.txt          (IMG_S)
    outputImageIntensityStat_48Atlas.txt          (IMG_C)

If --group6 is used alone, the following files are expected:

    outputlabelGeometryMeasures_115Label_AFF.txt  (AFF)
    outputlabelGeometryMeasures_115Label_PHY.txt  (PHY)
    outputimageIntensityStat_115Atlas.txt         (IMG)

If no flag is used, the following files are expected:
    outputimageIntensityStatWarp_115Label_MNI*.txt   (MNI)
    outputlabelGeometryMeasures_115Label*.txt        (GEO)

    """
    parser = argparse.ArgumentParser(description='Parse data from CT registration', epilog=epi,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_dir', type=str, help='the input directory to use')
    parser.add_argument('output', type=str, help='the path to write the final DataFrame to')
    parser.add_argument('--group6', action='store_true', default=False, help='data is grouped into three age ranges and split by sex')
    parser.add_argument('--sep_atlas', action='store_true', default=False, help='cortical and subcortical measures are separate')

    args = parser.parse_args(argv)

    files = list()
    if args.group6:
        if args.sep_atlas:
            geom_cols = "Volume SurfArea Eccentricity Elongation Orientation".split(" ")
            files.append((f"{args.input_dir}/*/outputLabelGeometryMeasures_21Label_AFF.txt", "AFF_S", geom_cols))
            files.append((f"{args.input_dir}/*/outputLabelGeometryMeasures_48Label_AFF.txt", "AFF_C", geom_cols))
            files.append((f"{args.input_dir}/*/outputLabelGeometryMeasures_21Label_PHY.txt", "PHY_S", geom_cols))
            files.append((f"{args.input_dir}/*/outputLabelGeometryMeasures_48Label_PHY.txt", "PHY_C", geom_cols))

            img_cols = "Mean Sigma Skewness Kurtosis Entropy Sum Q05 Q95 Min Max".split(" ")
            files.append((f"{args.input_dir}/*/outputImageIntensityStat_21Atlas.txt", "IMG_S", img_cols))
            files.append((f"{args.input_dir}/*/outputImageIntensityStat_48Atlas.txt", "IMG_C", img_cols))
        else:
            geom_cols = "Volume SurfArea Eccentricity Elongation Orientation".split(" ")
            files.append((f"{args.input_dir}/*/outputlabelGeometryMeasures_115Label_AFF.txt", "AFF", geom_cols))
            files.append((f"{args.input_dir}/*/outputlabelGeometryMeasures_115Label_PHY.txt", "PHY", geom_cols))

            img_cols = "Mean Sigma Skewness Kurtosis Entropy Sum Q05 Q95 Min Max".split(" ")
            files.append((f"{args.input_dir}/*/outputimageIntensityStat_115Atlas.txt", "IMG", img_cols))
    else:
        img_cols = "Mean Sigma Skewness Kurtosis Entropy Sum Q05 Q95 Min Max".split(" ")
        files.append((f"{args.input_dir}/*/outputimageIntensityStatWarp_115Label_MNI*.txt", "MNI", img_cols))

        geom_cols = "Volume SurfArea Eccentricity Elongation Orientation".split(" ")
        files.append((f"{args.input_dir}/*/outputlabelGeometryMeasures_115Label*.txt", "GEO", geom_cols))

    df = read_all(*files)
    df.to_csv(args.output)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
