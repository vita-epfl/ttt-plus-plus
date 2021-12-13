import sys
sys.path.append("..")
import torch
import numpy as np
from discrepancy import *

def visu_feat(encoder, dataloader, figname, num_sample=9216):
    encoder.eval()
    if dataloader.batch_size >= num_sample:
        num_batch = 1
    else:
        num_batch, mod = divmod(num_sample, dataloader.batch_size)
        assert mod == 0, "Batch size error"
    stack_feat = list()
    stack_label = list()
    dl_iter = iter(dataloader)
    with torch.no_grad():
        for _ in range(num_batch):
            inputs, labels = next(dl_iter)
            features = encoder(inputs.cuda())
            stack_feat.append(features.cpu().numpy())
            stack_label.append(labels.numpy())
    features_concat = np.concatenate(stack_feat)
    labels_concat = np.concatenate(stack_label)
    features_tsne_concat = feat_tsne(features_concat, labels_concat, figname)
    print('Save feature visualization to', figname)
    return features_concat, labels_concat, features_tsne_concat

def comp_feat(feat_src, label_src, feat_tar, label_tar, figname):
    features_concat = np.concatenate([feat_src, feat_tar])
    label_src[:] = 0
    label_tar[:] = 1
    labels_concat = np.stack([label_src, label_tar])
    feat_tsne(features_concat, labels_concat, figname)
    print('Save feature comparision to', figname)

def ext_param(ckpt_net):
    ckpt_ext = dict()
    for name in ckpt_net.keys():
        if name[:2] == 'co':
            key = '0' + name[5:]
        elif name[:2] == 'la':
            key = name[5:]
        elif name[:2] == 'bn':
            key = '4' + name[2:]
        else:
            continue
        ckpt_ext[key] = ckpt_net[name]
    print("Extract parameters of encoder...")
    return ckpt_ext

def ext_joint50_param(ckpt_net):
    ckpt_ext = dict()
    for name in ckpt_net.keys():
        if "encoder" in name:
            if name[:7] == 'encoder':
                key = name[8:]
            else:
                key = name[15:]
        else:
            continue
        ckpt_ext[key] = ckpt_net[name]
    print("Extract parameters of resnet50 encoder...")
    return ckpt_ext

def ext_bn50_param(ckpt_net):
    ckpt_ext = dict()
    for name in ckpt_net.keys():
        if 'bn' in name:
            key = name.replace("encoder.", "ext.")
        else:
            continue
        ckpt_ext[key] = ckpt_net[name]
    print("Extract parameters of BatchNorm for Tent...")
    return ckpt_ext

def calculate_distance(feat_src, label_src, tsne_src, feat_tar, label_tar, tsne_tar):
    L_feat_tot = 0
    L_tsne_tot = 0

    from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

    # Define a Sinkhorn (~Wasserstein) loss between sampled measures
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05, scaling=0.8)

    for class_id in range(10):
        feat_src_class = torch.from_numpy(feat_src[label_src == class_id])
        tsne_src_class = torch.from_numpy(tsne_src[label_src == class_id])

        feat_tar_class = torch.from_numpy(feat_tar[label_tar == class_id])
        tsne_tar_class = torch.from_numpy(tsne_tar[label_tar == class_id])

        L_feat = loss(feat_src_class, feat_tar_class)
        L_tsne = loss(tsne_src_class, tsne_tar_class)
        
        L_feat_tot += L_feat
        L_tsne_tot += L_tsne
        print("Class, Loss_feat, Loss_tsne: ", class_id, L_feat, L_tsne)

    L_feat_tot /= 10
    L_tsne_tot /= 10
    print("Loss_feat, Loss_tsne: ", L_feat_tot, L_tsne_tot)
