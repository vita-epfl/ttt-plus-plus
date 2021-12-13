import os
import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import seaborn as sns
sns.set_theme()

import argparse

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

########################################################################

def parse_arguments():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--foldername', type=str, default='results/cifar10_test_layer3_gn_expand')
    args = parser.parse_args()
    return args


def gather_test(foldername):
    summary = []

    common_corruptions.sort()

    for corruption in common_corruptions:
        fname = os.path.join(foldername, corruption + '_test.csv')
        df = pd.read_csv(fname, usecols=[0, 2, 4], names=["Epoch", "cls", "ssl"])
        df["Corruption"] = corruption
        summary.append(df)
    
    df = pd.read_csv(fname, usecols=[0, 1, 3], names=["Epoch", "cls", "ssl"])
    df["Corruption"] = 'original'
    summary.append(df)

    df = pd.concat(summary)

    return df


def plot_errors(df, foldername):
    
    df["cls"] *= 100.0
    clsplot = sns.lineplot(data=df, x="Epoch", y="cls", hue="Corruption", palette="husl")
    clsplot.set(ylim=(0.0, 80.0))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    figname = os.path.join(foldername, 'cls.png')
    plt.savefig(figname, bbox_inches='tight', pad_inches=0)
    print('Save figure to {}'.format(figname))

    plt.clf()

    df["ssl"] *= 100.0
    sslplot = sns.lineplot(data=df, x="Epoch", y="ssl", hue="Corruption", palette="husl")
    sslplot.set(ylim=(0.0, 80.0))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    figname = os.path.join(foldername, 'ssl.png')
    plt.savefig(figname, bbox_inches='tight', pad_inches=0)
    print('Save figure to {}'.format(figname))


def main():
    args = parse_arguments()
    df = gather_test(args.foldername)
    plot_errors(df, args.foldername)


if __name__ == '__main__':
    main()
