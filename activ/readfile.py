from argparse import ArgumentTypeError, ArgumentParser
import h5py as _h5py
import numpy as _np
from warnings import warn
from pkg_resources import resource_filename



class TrackTBIFile(object):

    __strtype = _h5py.special_dtype(vlen=bytes)

    __bm = 'biomarkers'
    __oc = 'outcomes'
    __bm_feat = 'biomarker_features'
    __oc_feat = 'outcome_features'
    __pid = 'patient_ids'

    def __init__(self, filename):
        self.filename = filename
        with _h5py.File(self.filename, 'r') as f:
            self.biomarkers = f[self.__bm][:]
            self.outcomes = f[self.__oc][:]
            self.biomarker_features = None
            self.outcome_features = None
            self.patient_ids = None
            if self.__bm_feat in f:
                self.biomarker_features = self.__decode(f[self.__bm_feat])
            if self.__oc_feat in f:
                self.outcome_features = self.__decode(f[self.__oc_feat])
            if self.__pid in f:
                self.patient_ids = self.__decode(f[self.__pid])

    def __decode(self, dset):
        if _h5py.check_dtype(vlen=dset.dtype) == bytes:
            return _np.array([s.decode('utf-8') for s in dset])
        else:
            return dset[:]

    @classmethod
    def __write_ascii(cls, grp, name, it):
        b = [bytes(x, 'utf-8') for x in it]
        ret = grp.create_dataset(name, data=b, dtype=cls.__strtype)
        return ret

    @classmethod
    def __add_dimscale(cls, dset, dim, scale, ann):
        if scale.attrs.get('CLASS', None) != 'DIMENSION_SCALE':
            dset.dims.create_scale(scale, ann)
        dset.dims[dim].attach_scale(scale)

    @classmethod
    def write(cls, dest, biomarkers, outcomes, biomarker_features=None, outcome_features=None, patient_ids=None):
        """
        Write biomarkers and outcomes to an HDF5 file

        Args:
            dest                : the path to an HDF5 file or the :class:`h5py.Group` to write to
            biomarkers          : the biomarker data with shape (n_samples, n_features)
            outcomes            : the outcomes data with shape (n_samples, n_features)
            biomarker_features  : the names of the biomarker features (optional)
            outcome_features    : the names of the outcome features (optional)
            patient_ids         : the patient IDs (optional)

        If biomarker_features, outcome_features, or patient_ids are provided, they will be
        added as dimension scales to the appropriate dimension of their respective datasets.
        """
        if biomarkers.shape[0] != outcomes.shape[0]:
            warn("biomarkers and outcomes do not have the same number of samples")
        h5group = dest
        close_grp = False
        if isinstance(dest, str):
            h5group = _h5py.File(dest, 'w')
            close_grp = True
        bm_dset = h5group.create_dataset(cls.__bm, data=biomarkers)
        oc_dset = h5group.create_dataset(cls.__oc, data=outcomes)

        # write biomarker feature names
        if biomarker_features is not None:
            if len(biomarker_features) != biomarkers.shape[1]:
                warn('biomarker_features length does not match biomarkers second dimension')
            scale = cls.__write_ascii(h5group, cls.__bm_feat, biomarker_features)
            cls.__add_dimscale(bm_dset, 1, scale, 'Biomarker feature names')

        # write outcome feature names
        if outcome_features is not None:
            if len(outcome_features) != outcomes.shape[1]:
                warn('outcome_features length does not match outcomes second dimension')
            scale = cls.__write_ascii(h5group, cls.__oc_feat, outcome_features)
            cls.__add_dimscale(oc_dset, 1, scale, 'Outcome feature names')

        # write patient IDs
        if patient_ids is not None:
            if len(patient_ids) != outcomes.shape[0]:
                print(len(patient_ids), outcomes.shape[1])
                warn('patient_ids length does not match outcomes first dimension')
            if len(patient_ids) != biomarkers.shape[0]:
                warn('patient_ids length does not match biomarkers first dimension')
            scale = cls.__write_ascii(h5group, cls.__pid, patient_ids)
            cls.__add_dimscale(oc_dset, 0, scale, 'Patient IDs')
            cls.__add_dimscale(bm_dset, 0, scale, 'Patient IDs')

        if close_grp:
            h5group.close()

class UoINMFTrackTBIFile(TrackTBIFile):

    __bm_bases = 'biomarker_bases'
    __oc_bases = 'outcome_bases'

    def __init__(self, filename):
        super(UoINMFTrackTBIFile, self).__init__(filename)
        with _h5py.File(self.filename, 'r') as f:
            self.biomarker_bases = f[self.__bm_bases][:]
            self.outcome_bases = f[self.__oc_bases][:]

    @classmethod
    def write(cls, dest, biomarkers, outcomes, biomarker_bases, outcome_bases,
              biomarker_features=None, outcome_features=None, patient_ids=None):
        """
        Write biomarkers and outcomes to an HDF5 file

        Args:
            dest                : the path to an HDF5 file or the :class:`h5py.Group` to write to
            biomarkers          : the biomarker data with shape (n_samples, n_features)
            outcomes            : the outcomes data with shape (n_samples, n_features)
            biomarker_bases     : the biomarker bases with shape (n_samples, n_features)
            outcome_bases       : the outcomes data with shape (n_samples, n_features)
            biomarker_features  : the names of the biomarker features (optional)
            outcome_features    : the names of the outcome features (optional)
            patient_ids         : the patient IDs (optional)

        If biomarker_features, outcome_features, or patient_ids are provided, they will be
        added as dimension scales to the appropriate dimension of their respective datasets.
        """
        h5group = dest
        close_grp = False
        if isinstance(dest, str):
            h5group = _h5py.File(dest, 'w')
            close_grp = True
        super(UoINMFTrackTBIFile, cls).write(h5group, biomarkers, outcomes,
                                             biomarker_features=biomarker_features,
                                             outcome_features=outcome_features,
                                             patient_ids=patient_ids)
        bm_dset = h5group.create_dataset(cls.__bm_bases, data=biomarker_bases)
        oc_dset = h5group.create_dataset(cls.__oc_bases, data=outcome_bases)
        if close_grp:
            h5group.close()


def read_file(filename):
    try:
        readfile_obj = TrackTBIFile(filename)
    except Exception as e:
        raise ArgumentTypeError(str(e))
    return readfile_obj

def impute(df):
    '''
    Fill missing values (as encoded by NaN).

    Missing categorical data (i.e. string-valued columns) and integer-valued
    columns will be filled with the most frequent value in the column and
    continuous (i.e. float-valued columns) will be filled with the mean of
    the values that exist in the column.
    '''
    import pandas as pd
    cols = dict()
    for colname, coltype in df.dtypes.items():
        fillval = None
        if coltype == object:
            fillval = df[colname].dropna().value_counts().idxmax()
        elif 'int' in coltype.name:
            fillval = df[colname].dropna().value_counts().idxmax()
        elif 'float' in coltype.name:
            fillval = df[colname].dropna().mean()
        cols[colname] = df[colname].fillna(fillval)
    newdf = pd.DataFrame(cols, index=df.index)
    return newdf

def encode(df, get_binvars=False):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    cols = dict()
    enc = LabelEncoder()
    lb = LabelBinarizer()
    binvars = dict()
    for colname, coltype in df.dtypes.items():
        if coltype == object:
            newcol = enc.fit_transform(df[colname])
            if len(enc.classes_) > 2:
                newcols = lb.fit_transform(newcol)
                for i, c in enumerate(newcols.T):
                    name = "%s=%s" % (colname, enc.classes_[lb.classes_[i]])
                    cols[name] = c
            else:
                cols[colname] = newcol
                binvars[colname] = enc.classes_
        else:
            cols[colname] = df[colname]
    newdf = pd.DataFrame(cols, index=df.index)
    if get_binvars:
        return newdf, binvars
    else:
        return newdf

def filter_invariant(df, get_inv_col=False):
    import numpy as np
    variable = np.std(df, axis=0) > 0
    cols = np.where(variable)[0]
    ret = df.iloc[:,cols]
    if get_inv_col:
        return ret, np.where(np.logical_not(variable))[0]
    else:
        return ret

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


def load_data(uoinmf=False):
    """
    Load a TRACK-TBI dataset.

    Use load_data('uoinmf') to laod UoI-NMF dataset

    Args:
        uoinmf  : True to load the UoINMF decomposed data
    """
    if uoinmf:
        path = resource_filename(__name__, 'data_uoinmf.h5')
        return UoINMFTrackTBIFile(path)
    else:
        path = resource_filename(__name__, 'data.h5')
        return TrackTBIFile(path)


# 586 samples with 474 features (258 biomarkers and 216 outcomes)


def load_rowmask():
    """
    Return a boolean mask for the rows to keep after
    filtering using Ju Han's occupancy clustering heatmap approach
    """
    import numpy as np
    nrow = 586
    row = resource_filename(__name__, 'keep_rows.txt')
    row = np.loadtxt(row, dtype=int)
    tmp = np.zeros(nrow, dtype=bool)
    tmp[row] = True
    return tmp

def load_colmask():
    """
    Return a boolean mask for the columns to keep after
    filtering using Ju Han's occupancy clustering heatmap approach
    """
    import numpy as np
    ncol = 918
    col_rm_path = resource_filename(__name__, 'keep_cols.txt')
    col_rm = np.loadtxt(col_rm_path, dtype=int)
    tmp = np.zeros(ncol, dtype=bool)
    tmp[col_rm] = True
    return tmp

def load_preprocessed(trim_col=False, impute_missing=False):
    from pandas import read_csv
    path = resource_filename(__name__, 'preprocessed_data_matrix.csv')
    ret = read_csv(path, index_col=0)
    if trim_col:
        cols = load_colmask()
        ret = ret.iloc[:,cols]
    if impute_missing:
        ret = impute(ret)
    return ret


def load_outcome_mask(trim_col=False):
    """
    Return a boolean mask for the columns in the preprocessed data
    that are outcomes.

    Use this to separate outcomes and biomarkers
    """
    import numpy as np
    path = resource_filename(__name__, 'mask_feature_outcome.txt')
    ret = np.loadtxt(path, dtype=bool)
    if trim_col:
        cols = load_colmask()
        ret = ret[cols]
    return ret
