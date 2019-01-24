import h5py
from activ import load_data
from activ.readfile import UoINMFTrackTBIFile


def read(path, name):
    f = h5py.File(path, 'r')
    d = f[name][:]
    bases = f["%s_bases" % name][:]
    f.close()
    return (d, bases)

parser = argparse.ArgumentParser(usage="%(prog)s biomarkers_h5 outcomes_h5 output_h5",
                                 formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))

parser.add_argument("biomarkers_h5", type=str, help="the UoINMF biomarkers output")
parser.add_argument("outcomes_h5", type=str, help="the UoINMF outcomes output")
parser.add_argument("output_h5", type=str, help="the path to the HDF5 file to save results to")

args = parser.parse_args()

d = load_data()

bm, bm_bases = read(args.biomarkers_h5)
oc, oc_bases = read(args.outcomes_h5)

f = h5py.File(args.output_h5, 'w')
UoINMFTrackTBIFile.write(f, bm, oc, bm_bases, oc_bases,
                         biomarker_features=d.biomarker_features,
                         outcome_features=d.outcome_features,
                         patient_ids=d.patient_ids)
f.close()
