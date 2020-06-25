import matplotlib.pyplot as plt
import argparse
import os.path
from activ.clustering.summarize import read_clustering_results, plot_results


parser = argparse.ArgumentParser()
parser.add_argument('input', help='output from clustering pipeline')
parser.add_argument('-o', '--outdir', help='the directory to save plots to', default='.')
parser.add_argument('-S', '--smooth', help='smooth plotted lines', default=False, action='store_true')


args = parser.parse_args()

def fix_ticks(ylabel):
    tick_fs = 15
    lbl_fs = 20
    ax = plt.gca()

    # Set the Y ticks
    ticks = ax.get_yticks()[:-1]
    labels = [str(int(t)) for t in ticks*100]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=tick_fs)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    plt.ylabel(ylabel, fontsize=lbl_fs)

    # Set the X ticks
    ticks = ax.get_xticks()[1:-1]
    labels = [str(int(t)) for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=tick_fs)
    ax.xaxis.set_label_coords(0.5, -0.1)
    plt.xlabel('# Outcome clusters', fontsize=lbl_fs)

def output(fname):
    return os.path.join(args.outdir, fname)

noc, foc, acc, cha = read_clustering_results(args.input)

# plot fold over chance
plt.figure(figsize=(10,10))
plot_results(noc, foc, color='red', smooth=args.smooth)
fix_ticks('Accuracy (% of chance)')
plt.savefig(output('fold_over_chance.png'))

# plot accuracy and chance
plt.figure(figsize=(10,10))
plot_results(noc, acc, color='black', smooth=args.smooth)
plot_results(noc, cha, color='gray', smooth=args.smooth)
fix_ticks('Accuracy (% correct)')
plt.savefig(output('accuracy_chance.png'))
