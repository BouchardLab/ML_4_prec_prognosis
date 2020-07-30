def combine(df, domain=True, subdomain=True):
    if domain and subdomain:
        if len(df['sub-domain']) > 0:
            subdd = df['domain'] + ' - ' + df['sub-domain']
        else:
            subdd = df['domain']
    elif subdomain:
        subdd = df['sub-domain']
    else:
        subdd = df['domain']
    return subdd


def get_domain_weight(df, ddict, coef, domain=True, subdomain=True, scale=False):
    """

    Args:
        scale: scale weights by the number of variables in the domain
    """
    mask = coef != 0
    weights = coef[mask]
    weights = np.abs(weights)
    weights /= weights.sum()
    col = df.columns[mask]
    subdd = ddict.filter(items=col, axis=0)
    subdd = combine(subdd, domain, subdomain)
    dweight = dict()
    for d in np.unique(subdd):
        mask = subdd == d
        if scale:
            weights[mask.values]
            dweight[d] = weights[mask].mean()
        else:
            dweight[d] = weights[mask].sum()
    s = pd.Series(data = dweight)
    s /= s.sum()
    return s

import matplotlib.colors as mpc
def multi_stemplot(values, varclass=None, ax=None, labels=None, palette='Set1', luts=None):
    if isinstance(values, np.ndarray):
        if len(values.shape) == 1:
            values = values.reshape(1, -1)
    if ax is None:
        ax = plt.gca()

    n_plots = len(values)

    maxstem = np.max([np.max(np.abs(values[i])) for i in range(n_plots)])
    shift = maxstem*2.1
    offsets = shift * np.arange(n_plots)

    offset = 0.0
    for i in range(n_plots):
        classes = varclass[i]
        offset = offsets[i]

        order = np.argsort(classes)
        values[i] = values[i][order]
        classes = classes[order]

        for c in np.unique(classes):
            val = np.zeros(len(classes)) + offset
            val[classes==c] += values[i][classes==c]
            col = mpc.to_hex(luts[i][c])
            ax.stem(val, use_line_collection=True, basefmt=col, linefmt=col, markerfmt=' ', bottom=offset)
        ax.stem(np.zeros(len(classes))+offset, basefmt='black', linefmt='black', markerfmt=' ', use_line_collection=True, bottom=offset)
    pos = offsets
    ax.get_xaxis().set_ticks([])
    yaxis = ax.get_yaxis()
    yaxis.set_ticks(pos)
    ticklabels = np.arange(n_plots)
    if labels is not None:
        ticklabels = labels
    yaxis.set_ticklabels(ticklabels, fontsize=20)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sort(wedges):
    angles = np.array([(p.theta2 + p.theta1)/2 % 360 for p in wedges])
    order = np.argsort(angles)
    q1 = list()
    q2 = list()
    q3 = list()
    q4 = list()
    for i in order:
        p = wedges[i]
        ang = angles[i]
        if ang < 90:
            q1.append(i)
        elif ang < 180:
            q2.append(i)
        elif ang < 270:
            q3.append(i)
        else:
            q4.append(i)
    q2 = q2[::-1]
    q4 = q4[::-1]
    return [q1, q2, q3, q4]

def clean_label(label):
    d, s = [s.strip() for s in label.split(' - ')]
    if len(s) == 0:
        return d
    else:
        return s

def donut_chart(ax, val, colors, radius=0.7):
    df_vals = pd.DataFrame.from_dict(val, orient='index')
    labels = df_vals.index.tolist()
    labels = [clean_label(l) for l in labels]
    wedges, texts = plt.pie(df_vals.values.squeeze(), counterclock = True, radius=radius,
                           startangle=90, colors=colors,
                           textprops={'fontsize': 14})
    bbox_props = dict(boxstyle="square,pad=0", fc="w", ec="w", lw=0.72)
    base_kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    for quad in sort(wedges):
        prev = None
        n_lines = 0
        for i in quad:
            p = wedges[i]
            ang = (p.theta2 + p.theta1)/2.# + p.theta1
            ang_wid = abs(p.theta2 - p.theta1)
            y = np.sin(np.deg2rad(ang)) * radius
            x = np.cos(np.deg2rad(ang)) * radius
            xytext = ((i/10) + 1.*np.sign(x), (i/10) + y)
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            xy  = (texts[i]._x, texts[i]._y)
            xytext = None
            kw = base_kw.copy()
            if (prev is not None and np.abs(prev[1] - xy[1]) < 0.5) or ang_wid < 10.0:
                connectionstyle = "angle,angleA=0,angleB={}".format(int(ang))
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                addl_rad = 1.2+0.1*n_lines
                xytext=(addl_rad*x, addl_rad*y)
                xy=(x, y)
                n_lines += 1
            ax.annotate(labels[i], xy=xy, xytext=xytext,
                        horizontalalignment=horizontalalignment, **kw, size=14)
            prev = xytext or xy

import seaborn as sns

pred_dd = datadict_df.filter(predictor_df.columns, axis=0)
uniq_pred = np.unique(combine(pred_dd))

pal = sns.color_palette('Set3', len(uniq_pred))
pal[1] = mpc.to_rgb('brown')
pred_pal_lut = dict(zip(uniq_pred, pal))

ocdd = datadict_df.filter(ocdf.columns, axis=0)
uniq_oc = np.unique(combine(ocdd))
pal = sns.color_palette('Set3', len(uniq_oc))
pal[1] = mpc.to_rgb('brown')
oc_pal_lut = dict(zip(uniq_oc, pal))

pred_class = combine(datadict_df).filter(predictor_df.columns, axis=0)
oc_class = combine(datadict_df).filter(ocdf.columns, axis=0)
scale = True

for i in range(k):
#for i in range(1):
    x_dweight = get_domain_weight(predictor_df, datadict_df, talscca.X_components_[i], scale=scale)
    y_dweight = get_domain_weight(ocdf, datadict_df, talscca.Y_components_[i], scale=scale)
    x_dweight.name = '' #'CV-%d predictors' % (i+1)
    y_dweight.name = '' #'CV-%d outcomes' % (i+1)
    plt.figure(figsize=(24, 20))
    ax = plt.subplot(2, 2, 4)
    xcolors = [pred_pal_lut[_] for _ in x_dweight.index]
    donut_chart(ax, x_dweight.to_dict(), xcolors)
    plt.title('Biomarkers & CT measures', fontsize=20)

    ax = plt.subplot(2, 2, 2)
    x = cv_bm[:,i]
    y = cv_oc[:, i]
    title = "%0.3f, %0.3f" % (x.var(), y.var())
    title = 'Canonical Variates'
    gcs_hurd = np.zeros(len(gcs_simple), dtype='U8')
    gcs_hurd[gcs_simple == 0.0] = 'Mild'
    gcs_hurd[gcs_simple == 1.0] = 'Moderate'
    gcs_hurd[gcs_simple == 2.0] = 'Severe'
    cross_decomp_scatter(x, y, fontsize=16, labels=gcs_hurd, title=title)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0)/abs(y1-y0))


    ax = plt.subplot(2, 2, 1)
    ycolors = [oc_pal_lut[_] for _ in y_dweight.index]
    donut_chart(ax, y_dweight.to_dict(), ycolors)
    plt.title('Outcomes', fontsize=20)
    #plt.tight_layout()

    plt.subplot(2, 2, 3)
    multi_stemplot([talscca.X_components_[i],talscca.Y_components_[i]],
                   varclass=[pred_class, oc_class], luts=[pred_pal_lut, oc_pal_lut],
                   labels=['Biomarkers &\nCT measures', 'Outcomes'])

    plt.title('Loadings', fontsize=20)
    #plt.tight_layout()
    plt.savefig('cca/cv%02d.png' % (i+1))
    plt.savefig('cca/cv%02d.pdf' % (i+1))
