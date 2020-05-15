import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize as _normalize

def load_data(tbi_measures_path, orig_tbi_data_path, filter_suffixes=('Sum', 'Min', 'Max', 'Q05', 'Q95'), normalize=True):
    """
    Args:
        normalize:        normalize patients within group
    """

    sdf = pd.read_csv(orig_tbi_data_path, header=0, index_col=0)
    df = pd.read_csv(tbi_measures_path, index_col=0)

    ptype = df['PatientType']
    measures = df.drop('PatientType', axis=1).columns

    df_patients = set(df.index)
    sdf_patients = set(sdf.index)
    bad_patients = list(df_patients ^ sdf_patients)
    df = df.drop(df_patients - sdf_patients, axis=0).sort_index(axis='index')
    sdf = sdf.drop(sdf_patients - df_patients, axis=0).sort_index(axis='index')

    gcs = sdf['GCSMildModSevereRecode'][:]
    gcs[np.where(gcs == ' ')[0]] = 'unmeasurable'

    todrop = list()
    for col in df.columns:
        if any(f in col for f in filter_suffixes):
            todrop.append(col)
    todrop.append('PatientType')
    Xdf = df.drop(todrop, axis=1)

    # check for NaNs
    X = Xdf.values
    bad_samples = np.unique(np.where(np.isnan(X))[0])
    good_samples = np.ones(X.shape[0], dtype=bool)
    good_samples[bad_samples] = False
    if bad_samples.shape[0] > 0:
        bad_patients = " ".join(ptype[bad_samples])
        raise ValueError(f'the following patients contains NaNs: {bad_patients}')

    if normalize:
        X_norm = Xdf.values.copy()
        #categories = np.unique(ptype)
        #for cat in categories:
        #    _mask = ptype==cat
        #    X_norm[_mask] = scale(X[_mask])
        #sex = pd.Series([p[0] for p in ptype])
        #sex_cat = np.unique(sex)
        #for cat in sex_cat:
        #    _mask = sex==cat
        #    X_norm[_mask] = scale(X[_mask])
        #X_norm = _normalize(X_norm)
        X_norm = scale(X_norm)
        Xdf = pd.DataFrame(data=X_norm, columns=Xdf.columns, index=Xdf.index)

    metadata_df = pd.concat([ptype, gcs], axis=1, sort=True)

    if not np.all(np.equal(Xdf.index, metadata_df.index)):
        raise ValueError('Something went wrong -- different patients in measure and GCS data')

    return Xdf, metadata_df


def get_age_sex(ptype):
    categories = np.unique(ptype)
    print(categories)
    # extract sex data
    sex = np.chararray(ptype.shape[0])
    sex[:] = 'M'
    for v in categories[:8]:
        sex[ptype==v] = 'F'
    sex = sex.astype('U')

    # extract age data
    age = np.zeros(ptype.shape[0], dtype='<U10')
#    age_cat = ['16-30', '31-50', '51-100']
#    age[:] = age_cat[0]
#    for v in categories[1::3]:
#        age[ptype==v] = age_cat[1]
#    for v in categories[2::3]:
#        age[ptype==v] = age_cat[2]
    age_cat = ['16-30', '31-40', '41-50' , '51-60', '61-70', '71-80', '81-90', '91-100']
    age[:] = age_cat[-1]
    for v in categories[range(14)[0::7]]:
        age[ptype==v] = age_cat[0]
    for v in categories[1::7]:
        age[ptype==v] = age_cat[1]
    for v in categories[2::7]:
        age[ptype==v] = age_cat[2]
    for v in categories[3::7]:
        age[ptype==v] = age_cat[3]
    for v in categories[4::7]:
        age[ptype==v] = age_cat[4]
    for v in categories[5::7]:
        age[ptype==v] = age_cat[5]
    for v in categories[6::7]:
        age[ptype==v] = age_cat[6]

    age = age.astype('U')

    return age, sex


def _check_pca(X, pca):
    if pca is None:
        if X is None:
            raise ValueError("X or pca must be provided")
        pca = PCA()
        pca.fit(X)
    return pca


def _check_path(path, **kwargs):
    if path is not None:
        plt.savefig(path, **kwargs)


def plot_overall_leverage(X=None, pca=None, var_thresh=0.88, path=None, **sf_kwargs):
    pca = _check_pca(X, pca)

    expvar = np.zeros(pca.n_components_ + 1)
    expvar[1:] = np.cumsum(pca.explained_variance_ratio_)
    ncomps = np.where(expvar > var_thresh)[0][0]
    comps = np.arange(expvar.shape[0])

    plt.subplot(1, 2, 1)
    plt.plot(comps, expvar, c='black')
    plt.scatter([comps[ncomps]], [expvar[ncomps]], s=100, edgecolors='black', c='none')
    plt.text(comps[ncomps]+20, expvar[ncomps],
             "%d components capture\n%0.0f%% of variance" % (comps[ncomps], 100*var_thresh),
             fontdict={'fontsize': 15}, verticalalignment='top')
    plt.xlabel("# of components", fontsize=15)
    plt.ylabel("% variance", fontsize=15)

    lev = np.sum(np.abs(pca.components_), axis=0)
    plt.subplot(1, 2, 2)
    plt.hist(lev, bins=40)
    plt.ylabel('Leverage', fontsize=15)
    _check_path(path, **sf_kwargs)
    return pca


def top_components(X=None, pca=None, index=None, var_thresh=0.88):
    """
    """
    pca = _check_pca(X, pca)
    expvar = np.zeros(pca.n_components_ + 1)
    expvar[1:] = np.cumsum(pca.explained_variance_ratio_)
    ncomps = np.where(expvar > var_thresh)[0][0]
    comps = np.arange(expvar.shape[0])
    Xpca = pca.transform(X)[:,:comps[ncomps]]
    pca_df = pd.DataFrame(data=Xpca, columns=["PC%03d" % i for i in range(ncomps)])
    return Xpca, pca_df

def plot_measure_leverage(measures, X=None, pca=None, palette='Set3', path=None, **sf_kwargs):
    """
    Args:
        measures:       the names of the CT measures
    """
    pca = _check_pca(X, pca)
    lev = np.sum(np.abs(pca.components_), axis=0)

    plt.subplot(1, 3, 1)
    mtype = pd.Series([m.split('_')[0] for m in measures])
    measure_types = np.unique(mtype)
    mtype_lev = np.zeros(measure_types.shape)
    for i, m in enumerate(measure_types):
        mtype_lev[i] = lev[mtype == m].mean()

    plt.pie(mtype_lev, labels=measure_types, colors=sns.color_palette(palette, mtype_lev.shape[0]), autopct='%1.1f%%',)
    plt.title('Registration Stage - Total Leverage', fontsize=15)

    plt.subplot(1, 3, 2)
    def get_mtype(m):
        ar = m.split('_')
        return ar[0] + '_' + ar[-2]
    mtype = pd.Series([get_mtype(m) for m in measures])
    measure_types = np.unique(mtype)
    mtype_lev = np.zeros(measure_types.shape)
    for i, m in enumerate(measure_types):
        mtype_lev[i] = lev[mtype == m].sum()

    plt.pie(mtype_lev, labels=measure_types, colors=sns.color_palette(palette, mtype_lev.shape[0]))
    plt.title('Measure - Total Leverage', fontsize=15)

    plt.subplot(1, 3, 3)

    def get_region(m):
        ar = m.split('_')
        if len(ar) == 4:
            return '_'.join(ar[1::2])
        else:
            return ar[-1]
    region = pd.Series([get_region(m) for m in measures])
    region_types = np.unique(region)
    region_lev = np.zeros(region_types.shape)
    for i, m in enumerate(region_types):
        region_lev[i] = lev[region == m].sum()

    plt.pie(region_lev, labels=region_types, colors=sns.color_palette(palette, region_lev.shape[0]))
    plt.title('Region - Total Leverage', fontsize=15)
    _check_path(path, **sf_kwargs)
    return pca


def plot_umap(emb, age, sex, path=None, sf_kwargs=dict(), sp_kwargs=dict()):
    """
    Args:
        emb:            UMAP embedding
        age:            age category of each patient
        sex:            sex of each patient
        path:           the path to save the plot to
        sf_kwargs:      keyword arguments for plt.savefig
        sp_kwargs:      keyword arguments for sns.scatterplot
    """
    sex_cat = np.sort(np.unique(sex))
    age_cat = np.sort(np.unique(age))

    sex_color = ['red', 'black']
    age_width = [2, 2, 2, 2, 2, 2, 2, 2]
    age_style = [':', '--', '-', '-.', (0, (1, 10)), (0, (1, 1)), (0, (5, 5)), (0, (5, 1))]

    x, y = emb[:,0], emb[:,1],

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    rect_corner = [left + width + spacing, bottom + height + spacing, 0.2, 0.2]

    # start with a rectangular Figure
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)

    ax_scatter.set_xlabel('1st UMAP dimension', fontsize=20)
    ax_scatter.set_ylabel('2nd UMAP dimension', fontsize=20)
    _sp_kwargs = dict(hue=sex,
                      palette=sns.color_palette(sex_color),
                      size=age, size_order=age_cat[::-1],
                      hue_order=sex_cat,
                      ax=ax_scatter)
    _sp_kwargs.update(sp_kwargs)
    sns.scatterplot(emb[:,0], emb[:,1], **_sp_kwargs)

    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_corner = plt.axes(rect_corner)
    ax_corner.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plot_kde = True
    leg = list()
    for sex_i, _sex in enumerate(sex_cat):
        sex_mask = sex == _sex
        label = _sex
        leg.append(mlines.Line2D([], [], color=sex_color[sex_i], label=_sex))
        for age_i, _age in enumerate(age_cat):
            age_mask = age == _age
            mask = np.logical_and(sex_mask, age_mask)
            kde_kws = {'linestyle': age_style[age_i], 'linewidth': age_width[age_i]}
            dp_kws = dict(hist=False, color=sex_color[sex_i], kde_kws=kde_kws)
            sns.distplot(emb[mask, 0], ax=ax_histx, **dp_kws)
            sns.distplot(emb[mask, 1], ax=ax_histy, vertical=True, **dp_kws)

    for age_i, _age in list(enumerate(age_cat))[::-1]:
        leg.append(mlines.Line2D([], [], color='black', linestyle=age_style[age_i], label=age_cat[age_i]),)

    ax_corner.axis('off')
    ax_corner.legend(handles=leg, loc='lower left')
    _check_path(path, **sf_kwargs)


def plot_gcs(emb, gcs, legend='brief', path=None, **sf_kwargs):
    sns.scatterplot(emb[:,0], emb[:,1], hue=gcs, legend=legend,
                    palette=sns.color_palette(['blue', 'red', 'green', 'black']))
    plt.xlabel('1st UMAP dimension', fontsize=20)
    plt.ylabel('2nd UMAP dimension', fontsize=20)
    _check_path(path, **sf_kwargs)



if __name__ == '__main__':
    import argparse
    import os

    epi = """
    """
    parser = argparse.ArgumentParser(description='Parse data from CT registration', epilog=epi,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ct_measures_csv', type=str, help='CSV of CT measures')
    parser.add_argument('tracktbi_csv', type=str, help='CSV of original TRACK-TBI data')
    parser.add_argument('-o', '--outdir', type=str, default='.', help='the directory to save figures to')

    parser.add_argument('--original', action='store_true', default=False, help='input_dir is directory containing original inputs')

    args = parser.parse_args()
    df = load_data(args.ct_measures_csv, args.tracktbi_csv)

    if args.original:
        filtsuf = ('Sum', 'Min', 'Max', 'Q05', 'Q95')
    else:
        filtsuf = ('Sum', 'Min', 'Max', 'Q05', 'Q95', 'PHY')

    def _p(filename):
        return os.path.join(args.outdir, filename)


    Xdf, metadata_df = load_data(args.ct_measures_csv, args.tracktbi_csv, filter_suffixes=filtsuf, normalize=True)
    age, sex = get_age_sex(metadata_df['PatientType'])
    pca = _check_pca(Xdf.values, None)

    plt.figure(figsize=(15, 7.5))
    plot_overall_leverage(pca=pca, path=_p('overall_leverage.png'))


    plt.figure(figsize=(21.5, 7.5))
    plot_measure_leverage(pca=pca, measures=Xdf.columns, path=_p('measure_leverage.png'), dpi=300)

    Xpca, pca_df = top_components(Xdf, pca)

    from umap import UMAP
    umap = UMAP(n_components=2, min_dist=0.0, random_state=1000)
    emb = umap.fit_transform(Xdf.values)

    plt.figure(figsize=(8, 8))

    plot_umap(emb, age, sex, path=_p('umap_summary.png'), dpi=300)

    plt.figure(figsize=(7,7))

    plot_gcs(emb, metadata_df['GCSMildModSevereRecode'], path=_p('umap_summary_gcs.png'), dpi=300)
