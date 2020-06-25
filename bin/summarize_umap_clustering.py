import matplotlib.pyplot as plt
import argparse
import os.path
from activ.clustering.summarize import read_clustering_results, plot_results


parser = argparse.ArgumentParser()
parser.add_argument('input', help='output from clustering pipeline')
parser.add_argument('-o', '--outdir', help='the directory to save plots to', default='.')
parser.add_argument('-S', '--smooth', help='smooth plotted lines', default=False, action='store_true')


args = parser.parse_args()


def output(fname):
    return os.path.join(args.outdir, fname)

noc, foc, acc, cha = read_clustering_results(args.input)

plt.figure()
plot_results(noc, foc, color='red', smooth=args.smooth)
plt.savefig(output('fold_over_chance.png'))

plt.figure()
plot_results(noc, acc, color='black', smooth=args.smooth)
plot_results(noc, cha, color='gray', smooth=args.smooth)
plt.savefig(output('accuracy_chance.png'))
