from argparse import ArgumentTypeError, ArgumentParser
import h5py as h5py
import numpy as np
import pandas as pd
from warnings import warn
from pkg_resources import resource_filename

def write_clustering_input(dest, X, Y, X_features=None, Y_features=None, sample_ids=None):
    """
    Format data for running UMAP clustering pipeline

    Args:
        dest         : the path to an HDF5 file or the :class:`h5py.Group` to write to
        X            : the X data with shape (n_samples, n_features)
        Y            : the Y data with shape (n_samples, n_features)
        X_features   : the names of the X features (optional)
        Y_features   : the names of the Y features (optional)
        sample_ids   : the samplel IDs (optional)

    If X_features, Y_features, or sample_ids are provided, they will be
    added as dimension scales to the appropriate dimension of their respective datasets.
    """
    return TrackTBIFile.write(dest, X, Y, biomarker_features=X_features, outcome_features=Y_features, patient_ids=sample_ids)


class Decomp(object):

    def __init__(self, bm, oc, bm_bases, oc_bases):
        self.bm = bm
        self.oc = oc
        self.bm_bases = bm_bases
        self.oc_bases = oc_bases

class TrackTBIFile(object):

    __strtype = h5py.special_dtype(vlen=str)

    __bm = 'biomarkers'
    __oc = 'outcomes'
    __bm_feat = 'biomarker_features'
    __oc_feat = 'outcome_features'
    __pid = 'patient_ids'

    __bm_feat_type = 'biomarker_type'
    __oc_feat_type = 'outcome_type'

    __cca = 'cca'
    __nmf = 'nmf'

    __bases = 'bases'

    __viz = 'viz'
    __bm_emb = 'biomarker_emb'
    __oc_emb = 'outcome_emb'
    __bm_col = 'biomarker_colors'
    __oc_col = 'outcome_colors'
    __bm_lbl = 'biomarker_labels'
    __oc_lbl = 'outcome_labels'

    @classmethod
    def bases(cls, d):
        return d + '_' + cls.__bases

    def __init__(self, filename, subgroup=None):
        idx = filename.rfind('.h5/')
        if idx > -1:
            idx += 3
            if subgroup is not None:
                warn('ignoring subgroup keyword argument, using '
                     f'{filename[idx:]} from filename argument instead')
            filename, subgroup = filename[:idx], filename[idx:]

        self.filename = filename
        with h5py.File(self.filename, 'r') as f:
            g = f
            if subgroup is not None:
                g = f[subgroup]
            self.biomarkers = g[self.__bm][:]
            self.outcomes = g[self.__oc][:]
            self.biomarker_features = self.__check_dset(g, self.__bm_feat, string=True)
            self.outcome_features = self.__check_dset(g, self.__oc_feat, string=True)
            self.patient_ids = self.__check_dset(g, self.__pid, string=True)
            self.biomarker_type = self.__check_feat_type(g, self.__bm_feat_type, self.biomarker_features)
            self.outcome_type = self.__check_feat_type(g, self.__oc_feat_type, self.outcome_features)
            self.nmf = self.__check_decomp(self.__nmf, g)
            self.cca = self.__check_decomp(self.__cca, g)
            if self.__viz in g:
                sg = g[self.__viz]
                self.biomarker_emb = self.__check_dset(sg, self.__bm_emb)
                self.outcome_emb = self.__check_dset(sg, self.__oc_emb)
                self.biomarker_colors = self.__check_dset(sg, self.__bm_col, string=True)
                self.outcome_colors = self.__check_dset(sg, self.__oc_col, string=True)
                self.biomarker_labels = self.__check_dset(sg, self.__bm_lbl, string=True)
                self.outcome_labels = self.__check_dset(sg, self.__oc_lbl, string=True)

        if self.biomarker_features is not None:
            idx = np.where(self.biomarker_features == 'GCSMildModSevereRecode')[0][0]
            gcs_simple = self.biomarkers[:, idx]
            self.gcs_simple = np.zeros(len(gcs_simple), dtype='U8')
            self.gcs_simple[gcs_simple == 0.0] = 'Mild'
            self.gcs_simple[gcs_simple == 1.0] = 'Moderate'
            self.gcs_simple[gcs_simple == 2.0] = 'Severe'
        else:
            self.gcs_simple = None

    def biomarker_df(self):
        return pd.DataFrame(data=self.biomarkers, columns=self.biomarker_features, index=self.patient_ids)

    def outcome_df(self):
        return pd.DataFrame(data=self.outcomes, columns=self.outcome_features, index=self.patient_ids)

    def datadict_df(self):
        return pd.concat([self.biomarker_type, self.outcome_type])

    def __check_decomp(self, decomp, grp):
        ret = None
        if decomp in grp:
            grp = grp[decomp]
            bm = grp[self.__bm][:]
            oc = grp[self.__oc][:]
            bm_bases = grp[self.bases(self.__bm)][:]
            oc_bases = grp[self.bases(self.__oc)][:]
            ret = Decomp(bm, oc, bm_bases, oc_bases)
        return ret

    def __check_feat_type(self, grp, feat_type, feat_names):
        ret = dict()
        if feat_type in grp:
            grp = grp[feat_type]
            for k in grp:
                ret[k] = self.__decode(grp[k])
        return pd.DataFrame(data=ret, index=feat_names)

    def __check_dset(self, grp, name, string=False):
        ret = None
        if name in grp:
            if string:
                ret = self.__decode(grp[name])
            else:
                ret = grp[name][:]
        return ret

    def __decode(self, dset):
        # if h5py.check_dtype(vlen=dset.dtype) == bytes:
        if h5py.check_dtype(vlen=dset.dtype) == str:
            return np.array([s.decode('utf-8') for s in dset])
        else:
            return dset[:]

    @classmethod
    def __write_str(cls, grp, name, it, overwrite=False):
        #b = [bytes(x, 'utf-8') for x in it]
        if name in grp and overwrite:
            del grp[name]
        dset = grp.create_dataset(name, shape=(len(it),), dtype=cls.__strtype)
        dset[:] = it
        return dset

    @classmethod
    def __write_dset(cls, grp, name, data, overwrite=False):
        if overwrite:
            if name in grp:
                del grp[name]
        return grp.create_dataset(name, data=data)

    @staticmethod
    def check_grp(h5group, mode):
        if isinstance(h5group, str):
            f = h5py.File(h5group, mode)
            return f, True
        return h5group, False

    @classmethod
    def __add_dimscale(cls, dset, dim, scale, ann):
        if scale.attrs.get('CLASS', None) != 'DIMENSION_SCALE':
            dset.dims.create_scale(scale, ann)
        dset.dims[dim].attach_scale(scale)

    @classmethod
    def write_viz(cls, h5group, bm_emb=None, oc_emb=None, bm_colors=None, oc_colors=None, bm_labels=None, oc_labels=None, overwrite=False):
        if bm_emb is oc_emb is bm_colors is oc_colors is None:
            return
        g = h5group.require_group(cls.__viz)
        if bm_emb is not None:
            cls.__write_dset(g, cls.__bm_emb, bm_emb, overwrite=overwrite)
        if oc_emb is not None:
            cls.__write_dset(g, cls.__oc_emb, oc_emb, overwrite=overwrite)
        if bm_colors is not None:
            cls.__write_dset(g, cls.__bm_col, bm_colors, overwrite=overwrite)
        if oc_colors is not None:
            cls.__write_dset(g, cls.__oc_col, oc_colors, overwrite=overwrite)
        if bm_labels is not None:
            cls.__write_str(g, cls.__bm_lbl, bm_labels, overwrite=overwrite)
        if oc_labels is not None:
            cls.__write_str(g, cls.__oc_lbl, oc_labels, overwrite=overwrite)

    @classmethod
    def write_nmf(cls, h5group, bm, oc, bm_bases, oc_bases, **kwargs):
        cls.__write_decomp(h5group, cls.__nmf, bm, oc, bm_bases, oc_bases, **kwargs)

    @classmethod
    def write_cca(cls, h5group, bm, oc, bm_bases, oc_bases, **kwargs):
        cls.__write_decomp(h5group, cls.__cca, bm, oc, bm_bases, oc_bases, **kwargs)

    @classmethod
    def __write_decomp(cls, h5group, name, bm, oc, bm_bases, oc_bases, metadata=None, overwrite=False):
        h5group, close = cls.check_grp(h5group, 'a')
        grp = h5group.require_group(name)
        pairs = [
            (cls.__bm, bm),
            (cls.__oc, oc),
            (cls.bases(cls.__bm), bm_bases),
            (cls.bases(cls.__oc), oc_bases),
        ]
        for name, data in pairs:
            cls.__write_dset(grp, name, data, overwrite=overwrite)

        if metadata is not None:
            for k, v in metadata.items():
                grp.attrs[k] = v
        if close:
            h5group.close()

    @classmethod
    def write_feat_types(cls, h5group, biomarkers, outcomes, **kwargs):
        # write biomarker feature names
        h5group, close = cls.check_grp(h5group, 'a')
        bmt_grp = h5group.require_group(cls.__bm_feat_type)
        biomarker_type = cls.check_type(biomarkers)
        for k, v in biomarker_type.items():
            scale = cls.__write_str(bmt_grp, k, v, **kwargs)

        # write outcome feature names
        oct_grp = h5group.require_group(cls.__oc_feat_type)
        outcome_type = cls.check_type(outcomes)
        for k, v in outcome_type.items():
            scale = cls.__write_str(oct_grp, k, v, **kwargs)
        if close:
            h5group.close()

    @classmethod
    def check_type(cls, feat_type):
        if isinstance(feat_type, dict):
            return feat_type
        return {'feat_type': feat_type}


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
        h5group, close_grp = cls.check_grp(dest, 'w')
        bm_dset = h5group.create_dataset(cls.__bm, data=biomarkers)
        oc_dset = h5group.create_dataset(cls.__oc, data=outcomes)

        # write biomarker feature names
        if biomarker_features is not None:
            if len(biomarker_features) != biomarkers.shape[1]:
                warn('biomarker_features length does not match biomarkers second dimension')
            scale = cls.__write_str(h5group, cls.__bm_feat, biomarker_features)

        # write outcome feature names
        if outcome_features is not None:
            if len(outcome_features) != outcomes.shape[1]:
                warn('outcome_features length does not match outcomes second dimension')
            scale = cls.__write_str(h5group, cls.__oc_feat, outcome_features)

        # write patient IDs
        if patient_ids is not None:
            if len(patient_ids) != outcomes.shape[0]:
                print(len(patient_ids), outcomes.shape[1])
                warn('patient_ids length does not match outcomes first dimension')
            if len(patient_ids) != biomarkers.shape[0]:
                warn('patient_ids length does not match biomarkers first dimension')
            scale = cls.__write_str(h5group, cls.__pid, patient_ids)

        if close_grp:
            h5group.close()

class UoINMFTrackTBIFile(TrackTBIFile):

    __bm_bases = 'biomarker_bases'
    __oc_bases = 'outcome_bases'

    def __init__(self, filename):
        super(UoINMFTrackTBIFile, self).__init__(filename)
        with h5py.File(self.filename, 'r') as f:
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
            h5group = h5py.File(dest, 'w')
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


def load_data(uoinmf=False, dead=False):
    """
    Load a TRACK-TBI dataset.

    Use load_data('uoinmf') to laod UoI-NMF dataset

    Args:
        uoinmf  : True to load the UoINMF decomposed data
        dead    : True to load dead patients imputed data
    """
    if uoinmf:
        path = resource_filename(__name__, 'data_uoinmf.h5')
        return UoINMFTrackTBIFile(path)
    elif dead:
        path = resource_filename(__name__, 'data_dead.h5')
        return TrackTBIFile(path)
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


def _clean_label(label):
    d, s = [s.strip() for s in label.split(' - ')]
    if len(s) == 0:
        return d
    else:
        return s


def get_feature_types(data_dict_path, bm_df, oc_df, ct_df=None, cntm_df=None):
    if data_dict_path is not None:
        # read from TRACK-TBI supplied data dict
        dd_df = pd.read_csv(data_dict_path, index_col=1)[['domain', 'sub-domain']].fillna(' ')

        # add any missing biomarker columns
        for c in set(bm_df.columns) - set(dd_df.index):
            sp = c.split('_')
            _c = sp[0]
            __c = '_'.join(sp[0:2])
            if _c in dd_df.index:
                # print('Adding missing biomarker', c)
                s = dd_df.loc[_c]
                s.name = c
                dd_df = dd_df.append(s)
            elif __c in dd_df.index:
                # print('Adding missing biomarker', c)
                s = dd_df.loc[__c]
                s.name = c
                dd_df = dd_df.append(s)

        # add any missing outcome columns
        for c in set(oc_df.columns) - set(dd_df.index):
            _c = '_'.join(c.split('_')[0:2])
            if _c in dd_df.index:
                # print('Adding missing outcome', c)
                s = dd_df.loc[_c]
                s.name = c
                dd_df = dd_df.append(s)
    else:
        # make up feature classes
        dd_df = pd.DataFrame(data={'domain': list(), 'sub-domain': list()})
        for c in bm_df.columns:
            dd_df = dd_df.append(pd.Series(name=c, data={'domain': 'Biomarker', 'sub-domain': ''}))
        for c in oc_df.columns:
            dd_df = dd_df.append(pd.Series(name=c, data={'domain': 'Outcome', 'sub-domain': ''}))

    if ct_df is not None:
        # add CT columns
        for c in ct_df.columns:
            dd_df = dd_df.append(pd.Series(name=c, data={'domain': 'Registered CT', 'sub-domain': 'MNI'}))
    if cntm_df is not None:
        # add CT columns
        for c in cntm_df.columns:
            dd_df = dd_df.append(pd.Series(name=c, data={'domain': 'Connectome', 'sub-domain': ''}))

    # add CT or connectome data if its been provided
    pred_colnames = list(bm_df.columns)
    if ct_df is not None:
        pred_colnames.extend(ct_df.columns)
    if cntm_df is not None:
        pred_colnames.extend(cntm_df.columns)

    # cleaned up data-dictionary values -- some features do not have a sub-domain
    cleaned_dd = [_clean_label(l) for l in (dd_df['domain'] + ' - ' + dd_df['sub-domain']).values]
    dd_df = pd.DataFrame(data=cleaned_dd, index=dd_df.index)

    # filter to order things appropriately
    bm_df, oc_df = dd_df.filter(pred_colnames, axis=0), dd_df.filter(oc_df.columns, axis=0)
    return bm_df, oc_df


def merge_data(scalar_data_path, output_path, ct_df=None, cntm_df=None, data_dict_path=None):
    """
    Args:
        scalar_data_path:     path to TRACK-TBI scalar data
        output_path:          path to save final data to
        ct_df:                CT data frame
        cntm_df:              connectome data frame
        data_dict_path:       the path to the data dictionary CSV from TRACK-TBI
    """
    tbi = TrackTBIFile(scalar_data_path)

    # read data into data frames for ease of merging
    bm_df = pd.DataFrame(data=tbi.biomarkers, index=tbi.patient_ids, columns=tbi.biomarker_features)
    pred_dfs = [bm_df]
    if ct_df is not None:
        pred_dfs.append(ct_df)
    if cntm_df is not None:
        pred_dfs.append(cntm_df)

    # merge data
    pred_df = pd.concat(pred_dfs, axis=1, join='inner') if len(pred_dfs) > 1 else pred_dfs[0]
    oc_df = pd.DataFrame(data=tbi.outcomes, index=tbi.patient_ids, columns=tbi.outcome_features).filter(pred_df.index, axis=0)

    # write merged data
    TrackTBIFile.write(output_path,
                    pred_df.values, oc_df.values,
                    biomarker_features=pred_df.columns,
                    outcome_features=oc_df.columns,
                    patient_ids=pred_df.index)

    # write feature types
    bm_ft, oc_ft = get_feature_types(data_dict_path, bm_df, oc_df, ct_df=ct_df, cntm_df=cntm_df)
    TrackTBIFile.write_feat_types(output_path, bm_ft.values.squeeze(), oc_ft.values.squeeze())
