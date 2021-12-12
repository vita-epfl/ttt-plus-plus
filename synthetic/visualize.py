"""System module."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
from sklearn.decomposition import PCA


def plot_data(x, y, figname):
    """plot dataset point"""
    fig, ax = plt.subplots()
    fig.set_size_inches(3, 3)
    cm_bright = colors.ListedColormap(['#FF0000', '#0000FF'])
    ax.scatter(x[:, 0], x[:, 1], s=2.0, c=y, alpha=0.5, cmap=cm_bright)
    ax.axis('square')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.savefig(figname, bbox_inches='tight')
    plt.close(fig)


def plot_prediction(x, y, net, scale, figname):
    """plot decision boundary"""
    fig, ax = plt.subplots()
    fig.set_size_inches(3, 3)
    cm_bright = colors.ListedColormap(['#FF0000', '#0000FF'])
    x0s = np.linspace(-scale, scale, 200)
    x1s = np.linspace(-scale, scale, 200)
    x0, x1 = np.meshgrid(x0s, x1s)
    xe = np.c_[x0.ravel(), x1.ravel()]
    xe = torch.from_numpy(xe).float()
    with torch.no_grad():
        y_pred, _, _ = net(xe)
    y_pred = y_pred.squeeze().gt(0.0)
    y_pred = y_pred.reshape(x0.shape).detach().numpy()
    ax.contourf(x0, x1, y_pred, cmap=cm_bright, alpha=0.05)
    ax.scatter(x[:, 0], x[:, 1], s=2.0, c=y, alpha=0.5, cmap=cm_bright)
    plt.axis('square')
    plt.xlim([-scale, scale])
    plt.ylim([-scale, scale])
    plt.savefig(figname, bbox_inches='tight')
    plt.close(fig)


def reduction(feat):
    """pca"""
    model = PCA(n_components=2)
    model.fit(feat)
    return model


def feat_tsne(embed, label, xlim, ylim, code=('#FF0000', '#0000FF'), figname=None):
    """plot feature alignment"""
    assert label.unique().size(0) <= 4

    tx, ty = embed[:, 0], embed[:, 1]

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 8)
    ax = fig.add_subplot(1, 1, 1)

    cmap = colors.ListedColormap(code)
    idx1 = label == 0
    idx2 = label == 1
    idx3 = label == 2
    idx4 = label == 3
    plt.scatter(tx[idx1], ty[idx1], c='r', cmap=cmap, alpha=0.8, s=50, marker='o', label='Source')
    plt.scatter(tx[idx2], ty[idx2], c='b', cmap=cmap, alpha=0.8, s=50, marker='o', label='Source')
    plt.scatter(tx[idx3], ty[idx3], c='r', cmap=cmap, alpha=0.8, s=50, marker='x', label='Target')
    plt.scatter(tx[idx4], ty[idx4], c='b', cmap=cmap, alpha=0.8, s=50, marker='x', label='Target')
    plt.legend()
    plt.axis('equal')
    plt.axis('off')

    plt.xlim(xlim)
    plt.ylim(ylim)

    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    plt.savefig(figname, bbox_inches='tight')
    plt.close(fig)
