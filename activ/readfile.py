from argparse import ArgumentTypeError, ArgumentParser


class TrackTBIFile(object):

    def __init__(self, filename):
        from h5py import File
        self.filename = filename
        with File(self.filename, 'r') as f:
            self.data_bm = f['data_matrix_subset_biomarker'][:]
            self.data_oc = f['data_matrix_subset_outcome'][:]
            self.feature_bm = f['feature_name_subset_biomarker'][:]
            self.feature_oc = f['feature_name_subset_outcome'][:]
            self.id = f['patient_id'][:]


def read_file(filename):
    try:
        readfile_obj = TrackTBIFile(filename)
    except Exception as e:
        raise ArgumentTypeError(str(e))
    return readfile_obj


def get_parser(usage="%(prog)s [options] filepath",
               desc="an analysis for TRACK-TBI data",
               epi=None):
    '''
    Return an argument parser where the first positional argument is
    a TRACK-TBI HDF5 file

    See argparse.ArgumentParser for more details on arguments

    Args:
        usage (str): the cmdline tool usage
        desc (str):  a description of what this tool does
        epi (str):   an epilog for this usage statement
    '''
    parser = ArgumentParser(usage=usage, description=desc, epilog=epi)
    parser.add_argument('filepath', type=read_file)
    return parser


def load_data():
    from pkg_resources import resource_filename
    path = resource_filename(__name__, 'data.h5')
    return TrackTBIFile(path)
