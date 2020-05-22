from activ.clustering import main, build_parser, get_noc
from activ.clustering.summarize import read_clustering_results


def test_subsample_umap_clustering():
    parser = build_parser()
    args = parser.parse_args(
        ['-d', 'activ/data.h5', '-p', 'activ/data.h5', '-S', 'subsample', '-i', '1', 'subsample.h5'])
    main(args)


def test_bootstrapped_umap_clustering():
    parser = build_parser()
    args = parser.parse_args(
        ['-d', 'activ/data.h5', '-p', 'activ/data.h5', '-S', 'bootstrap', '-i', '1', 'bootstrap.h5'])
    main(args)


def test_jacknifed_umap_clustering():
    parser = build_parser()
    args = parser.parse_args(
        ['-d', 'activ/data.h5', '-p', 'activ/data.h5', '-S', 'jackknife', '-i', '1', 'jackknife.h5'])
    main(args)
    _noc, _foc, _accuracy, _chance = read_clustering_results('jackknife.h5')
    kwargs = dict(noc=_noc,
                  foc=_foc,
                  plot=False,
                  fit_summary=True,
                  ttest_cutoff=False,
                  iqr=True,
                  use_median=True,
                  spread_asm=True,
                  spread_foc=False)
    get_noc(n_sigma=1.0, **kwargs)
    kwargs = dict(noc=_noc,
                  foc=_foc,
                  plot=False,
                  fit_summary=True,
                  ttest_cutoff=False,
                  iqr=True,
                  use_median=False,
                  spread_asm=True,
                  spread_foc=True)
    get_noc(n_sigma=1.0, **kwargs)
