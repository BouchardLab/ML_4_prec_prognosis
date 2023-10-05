#import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def draw_ellipse(x, y, width, height, ax=None, color='blue', linewidth=1, alpha=0.1):

    #ax = ax or plt.gca()
    face = Ellipse((x, y),
        width=width,
        height=height,
        facecolor=color,
        angle=45,
        alpha=alpha)
    edge = Ellipse((x, y),
        width=width,
        height=height,
        angle=45,
        edgecolor=color,
        facecolor='none',
        linewidth=linewidth)
    ax.add_patch(face)
    ax.add_patch(edge)


def get_cuts(left, right, num_circles):
    cuts = np.linspace(left, right, num_circles*2 + 1)
    center_idx = np.arange(num_circles*2 + 1)[1:-1:2]
    return cuts, center_idx


def add_circles(left, right, num_circles, color, num_levels=1, current_level=0, minor_d_frac=1.0, ax=None):
    if current_level >= num_levels:
        return
    #ax = ax or plt.gca()
    cuts, center_idx = get_cuts(left, right, num_circles)
    line_length = right-left
    d = np.sqrt(2) * line_length / num_circles
    for i, c_i in enumerate(center_idx):
        center = cuts[c_i]
        col = color
        if current_level == 0 and isinstance(color, (list, np.ndarray)):
            col = color[i]
        draw_ellipse(center, center, d, d*minor_d_frac, color=col, ax=ax, linewidth=0.5)
        add_circles(cuts[c_i-1], cuts[c_i+1], 2, col, num_levels, current_level+1, minor_d_frac=minor_d_frac, ax=ax)


def plot_cat2cont(ax=None, fontsize='x-large', marker_size=None):
    #ax = ax or plt.gca()

    major_d = 2
    minor_d = major_d/4

    ax.set_xlim(0, major_d)
    ax.set_ylim(0, major_d)

    ellipse_l = (major_d-np.sqrt(2))/2
    ellipse_r = major_d - ellipse_l

    draw_ellipse(1, 1, major_d*1.1, minor_d, color='black', linewidth=0.5, alpha=0.3, ax=ax)

    colors = ['blue', 'green', 'red']
    add_circles(ellipse_l, ellipse_r, 3, colors, 3, minor_d_frac=0.75, ax=ax)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    p_pos = 0.8
    p_lw = 4
    ax.axhline(p_pos, xmax=p_pos/major_d, linewidth=p_lw, c='k')
    ax.axvline(p_pos, ymax=p_pos/major_d, linewidth=p_lw, c='k')
    ax.scatter([p_pos], [p_pos], marker='*', color='yellow', edgecolor='black', s=marker_size, alpha=1.0, zorder=10, label="Individual patient")
    ax.legend(loc='upper left', fontsize=fontsize, frameon=False, borderaxespad=0.1)
    ax.set_xlabel('Intake features', fontsize=fontsize)
    ax.set_ylabel('Outcomes', fontsize=fontsize)
