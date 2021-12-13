import torch
import copy
import statistics

from discrepancy import *

def offline(trloader, ext, scale):
    ext.eval()

    mu_src = None
    cov_src = None

    coral_stack = []
    mmd_stack = []
    feat_stack = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(trloader):

            feat = ext(inputs.cuda())
            cov = covariance(feat)
            mu = feat.mean(dim=0)

            if cov_src is None:
                cov_src = cov
                mu_src = mu
            else:
                loss_coral = coral(cov_src, cov)
                loss_mmd = linear_mmd(mu_src, mu)
                coral_stack.append(loss_coral.item())
                mmd_stack.append(loss_mmd.item())
                feat_stack.append(feat)

    print("Source loss_mean: mu = {:.4f}, std = {:.4f}".format(scale, scale / statistics.mean(mmd_stack) * statistics.stdev(mmd_stack)))
    print("Source loss_coral: mu = {:.4f}, std = {:.4f}".format(scale, scale / statistics.mean(coral_stack) * statistics.stdev(coral_stack)))

    feat_all = torch.cat(feat_stack)
    feat_cov = covariance(feat_all)
    feat_mean = feat_all.mean(dim=0)
    return feat_cov, statistics.mean(coral_stack), feat_mean, statistics.mean(mmd_stack)
