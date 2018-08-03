import os
import logging
import sys
from datetime import datetime

from activ.readfile import TrackTBIFile
from activ.clustering import umap_cluster_sweep
from activ import load_data

if __name__ == '__main__':
    def int_list(string):
        if ':' in string:
            ar = [int(a) for a in string.split(":")]
            return list(range(ar[0], ar[1]+1))
        else:
            return list(map(int, string.split(",")))
    import argparse

    parser = argparse.ArgumentParser(usage="%(prog)s [options] output_h5",
                                     formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))

    parser.add_argument("output_h5", type=str, help="the path to the HDF5 file to save results to")
    parser.add_argument("-d", "--data", type=str, help="the Track TBI dataset file. use activ.load_data by default", default=None)
    parser.add_argument("-p", "--pdata", type=str, help="the Track TBI dataset file to use for predictions. use activ.load_data by default", default=None)
    parser.add_argument("-s", "--self_predict", help="predict clusters with dataset used to form clusters", action='store_true', default=False)
    parser.add_argument("-b", "--biomarkers", help="form clusters using biomarkers. use outcomes by default", action='store_true', default=False)
    parser.add_argument("-i", "--iters", type=int, help="the number of iterations to run", default=50)
    parser.add_argument("-u", "--umap_dims", type=int_list, help="a comma-separated list of the UMAP dimensions",
                        default=list(range(2, 11)))
    parser.add_argument("-c", "--cluster_size", type=int_list, help="a comma-separated list of the cluster sizes",
                        default=list(range(2, 15)))
    parser.add_argument("-f", "--force", help="force rerunning i.e. overwrite output_h5", action='store_true', default=False)
    parser.add_argument("-q", "--quiet", help="make log messages quiet", action='store_true', default=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    pargs = parser.parse_args()

    args = list()
    kwargs = dict()

    # set up logging
    logger = logging.getLogger('umap_cluster_sweep')
    logger.addHandler(logging.StreamHandler(sys.stderr))
    log_level = logging.INFO
    if pargs.quiet:
        log_level = logging.WARNING
    logger.setLevel(log_level)
    kwargs['logger'] = logger

    data = None         # source of data for building clusters
    pdata = None        # source of data for predicting cluster labels
    if pargs.data is None:
        data = load_data()
    else:
        data = TrackTBIFile(pargs.data)

    if pargs.pdata is None:
        pdata = data
    else:
        pdata = TrackTBIFile(pargs.pdata)

    args.append(pargs.iters)
    if pargs.biomarkers:
        logger.info('building clusters with biomarkers from %s' % data.filename)
        args.append(data.biomarkers)
        if pargs.self_predict:
            logger.info('predicting cluster labels with biomarkers from %s' % pdata.filename)
            kwargs['predict_data'] = pdata.biomarkers
        else:
            logger.info('predicting cluster labels with outcomes from %s' % pdata.filename)
            kwargs['predict_data'] = pdata.outcomes
    else:
        logger.info('building clusters with outcomes from %s' % data.filename)
        args.append(data.outcomes)
        if pargs.self_predict:
            logger.info('predicting cluster labels with outcomes from %s' % pdata.filename)
            kwargs['predict_data'] = pdata.outcomes
        else:
            logger.info('predicting cluster labels with biomarkers from %s' % pdata.filename)
            kwargs['predict_data'] = pdata.biomarkers

    args.append(pargs.umap_dims)
    args.append(pargs.cluster_size)

    path = pargs.output_h5
    if os.path.exists(path) and not pargs.force:
        sys.stderr.write("%s already exists. cautiously exiting. use -f to override\n" % path)
        sys.exit(255)
    kwargs['h5group'] = path
    start = datetime.now()
    logger.info("Begin UMAP pipeline at %s" % start.strftime('%Y-%m-%d %H:%M:%S'))
    res = umap_cluster_sweep(*args, **kwargs)
    end = datetime.now()
    logger.info("End UMAP pipeline at %s" % end.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("Time elapsed: %s" % str(end-start))
