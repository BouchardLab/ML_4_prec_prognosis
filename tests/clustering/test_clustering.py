from activ.clustering import main, build_parser


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
        ['-d', 'activ/data.h5', '-p', 'activ/data.h5', '-S', 'jackknife', '-i', '1', "jackknife.h5"])
    main(args)
