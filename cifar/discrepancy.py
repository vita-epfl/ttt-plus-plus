import numpy as np
import torch
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

def covariance(features):
    assert len(features.size()) == 2, "TODO: multi-dimensional feature map covariance"
    n = features.shape[0]
    tmp = torch.ones((1, n), device=features.device) @ features
    cov = (features.t() @ features - (tmp.t() @ tmp) / n) / (n - 1)
    return cov


def coral(cs, ct):
    d = cs.shape[0]
    loss = (cs - ct).pow(2).sum() / (4. * d ** 2)
    return loss


def linear_mmd(ms, mt):
    loss = (ms - mt).pow(2).mean()
    return loss


def feat_tsne(feat, label, figname):
    tsne = TSNE(n_components=2).fit_transform(feat)
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(1, 1, 1)

    num_class = len(np.unique(label))
    if num_class == 10:
        cmap = 'tab10'
    elif num_class == 2:
        cmap = colors.ListedColormap(['red', 'blue'])
    else:
        raise NotImplementedError

    plt.scatter(tx, ty, c=label, cmap=cmap, alpha=0.5)
    plt.axis('square')
    plt.axis('off')

    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    plt.savefig(figname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print('Save tsne to {}'.format(figname))

    return tsne
