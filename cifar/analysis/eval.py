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
import glob
import re

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

########################################################################

def parse_arguments():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--foldername', type=str, default='results/cifar10_tricks_layer3_gn_expand')
    parser.add_argument('--method', type=str, default=None, help='freeze, finetune')
    parser.add_argument('--terminate', type=str, default='thres', help='thres, optim')
    args = parser.parse_args()
    return args


def gather_domains(foldername, method):

    summary = []

    pattern = os.path.join(foldername, '*{}.csv'.format(method))
    files = glob.glob(pattern)

    for filename in files:
        df = pd.read_csv(filename, names=["CLS", "SSL"])
        df *= 100.0
        df = df.drop([0])           # first raw of original
        df['corruption'] = filename.split('/')[-1].split('_')[0]
        figname = filename[:-3] + 'png'
        summary.append(df)
    return pd.concat(summary)


def gather_method(foldername, terminate):

    summary = pd.DataFrame()

    pattern = os.path.join(foldername, '*.csv')
    files = glob.glob(pattern)

    files.sort()

    for filename in files:
        df = pd.read_csv(filename, names=["CLS", "SSL"])
        df *= 100.0

        corruption= filename.split('/')[-1].split('_')[0]
        
        config = re.split('_|\.', filename.split('/')[-1])

        # if config[-2] in ['freeze', 'finetune']:
        if config[-2] in ['freeze']:
            # method = config[-2]
            method = 'ssl'
        else:
            if config[-2] != '5000' or config[-3] != '1024':    # TODO:
                continue
            else:
                method = config[-4]

        err_cls_test = df["CLS"][1].item()
        summary = summary.append({'corruption': corruption,
                                    'method': 'test',
                                    'cls': err_cls_test,
                                    }, ignore_index=True)

        if terminate == 'thres':
            idx_thres = (df.iloc[1:, 1] - df["SSL"][0].item()).abs().argmin()
            err_cls_thres = df["CLS"][idx_thres + 1].item()
            summary = summary.append({'corruption': corruption,
                                        'method': method,
                                        'cls': err_cls_thres,
                                        }, ignore_index=True)
        elif terminate == 'optim':
            idx_optim = df.iloc[1:, 0].argmin()
            err_cls_optim = df["CLS"][idx_optim + 1].item()
            summary = summary.append({'corruption': corruption,
                                        'method': method,
                                        'cls': err_cls_optim,
                                        }, ignore_index=True)
        else:
            raise NotImplementedError

    summary.drop_duplicates(inplace=True, subset=['corruption', 'method'], keep='first')

    summary = summary.append({'corruption': 'original',
                                'method': 'test',
                                'cls': df["CLS"][0].item(),
                                }, ignore_index=True)

    # print(summary.head(10))
    # import pdb; pdb.set_trace()

    return summary


def plot_evolution(df, figname, vmax=60.0):
    plt.figure(figsize=(5, 6))
    plt.subplot(1, 1, 1)
    snsplot = sns.lineplot(data=df, x='SSL', y='CLS', hue='corruption', marker="o", palette="husl")
    snsplot.set(xlim=(0.0, vmax), ylim=(0.0, vmax))
    plt.axis('square')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(figname, bbox_inches='tight', pad_inches=0)
    print('Save figure to {}'.format(figname))


def plot_method(df, figname, vmax=60.0):
    snsplot = sns.catplot(data=df, x="corruption", y="cls", 
                        hue="method", kind="bar", aspect=2.8,
                        hue_order=["test", "ssl", "coral", "both"])
    snsplot.set(ylim=(0.0, vmax))
    plt.savefig(figname, bbox_inches='tight', pad_inches=0)
    print('Save figure to {}'.format(figname))


def stat_method(df):
    df_corruptions = df[(df["corruption"] != "original") & (df["corruption"] != "recover")]
    print("ssl: {:.2f}".format(df_corruptions[df_corruptions["method"] == "ssl"]["cls"].mean()))
    print("coral: {:.2f}".format(df_corruptions[df_corruptions["method"] == "coral"]["cls"].mean()))
    print("both: {:.2f}".format(df_corruptions[df_corruptions["method"] == "both"]["cls"].mean()))

    df_recovery = df[df["corruption"] == "recover"]
    print(df_recovery)

    df_original = df[df["corruption"] == "original"]
    print(df_original)

def main():
    args = parse_arguments()
    # print(args)

    if args.method is None:
        df = gather_method(args.foldername, args.terminate)
        plot_method(df, os.path.join(args.foldername, args.terminate + '.png'))
        stat_method(df)
    else:
        df = gather_domains(args.foldername, args.method)
        plot_evolution(df, os.path.join(args.foldername, args.method + '.png'))


if __name__ == '__main__':
    main()
