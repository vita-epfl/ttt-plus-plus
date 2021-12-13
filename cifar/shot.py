from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.misc import *
from utils.test_helpers import *
from utils.prepare_dataset import *

# ----------------------------------

import copy
import time
import pandas as pd

import random
import numpy as np

from discrepancy import *
from offline import *
from utils.trick_helpers import *
from utils.contrastive import *

from utils.shot_utils import obtain_shot_label, Entropy
from utils.shot_utils import op_copy, lr_scheduler
# ----------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--dataroot', default=None)
parser.add_argument('--shared', default=None)
########################################################################
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--group_norm', default=0, type=int)
parser.add_argument('--workers', default=0, type=int)
parser.add_argument('--num_sample', default=1000000, type=int)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--nepoch', default=500, type=int, help='maximum number of epoch for ttt')
parser.add_argument('--bnepoch', default=2, type=int, help='first few epochs to update bn stat')
parser.add_argument('--delayepoch', default=0, type=int)
parser.add_argument('--stopepoch', default=25, type=int)
########################################################################
parser.add_argument('--outf', default='.')
########################################################################
parser.add_argument('--level', default=5, type=int)
parser.add_argument('--corruption', default='snow')
parser.add_argument('--resume', default=None, help='directory of pretrained model')
parser.add_argument('--ckpt', default=None, type=int)
parser.add_argument('--fix_ssh', action='store_true')
########################################################################
parser.add_argument('--method', default='shot', choices=['shot'])
########################################################################
parser.add_argument('--model', default='resnet50', help='resnet50')
parser.add_argument('--save_every', default=100, type=int)
########################################################################
parser.add_argument('--tsne', action='store_true')
########################################################################
parser.add_argument('--cls_par', type=float, default=0.001)
parser.add_argument('--ent_par', type=float, default=1.0)
parser.add_argument('--gent', type=bool, default=True)
parser.add_argument('--ent', type=bool, default=True)
########################################################################
parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()

print(args)

my_makedir(args.outf)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# -------------------------------

net, ext, head, ssh, classifier = build_resnet50(args)

_, teloader = prepare_test_data(args)

# -------------------------------

args.batch_size = min(args.batch_size, args.num_sample)
_, trloader = prepare_test_data(args, num_sample=args.num_sample)

# -------------------------------

print('Resuming from %s...' %(args.resume))

load_resnet50(net, head, ssh, classifier, args)

## SHOT (turn off classifier grad)
for k, v in classifier.named_parameters():
    v.requires_grad = False

if torch.cuda.device_count() > 1:
    ext = torch.nn.DataParallel(ext)

# ----------- Test ------------

if args.tsne:
    args_src = copy.deepcopy(args)
    args_src.corruption = 'original'
    _, srcloader = prepare_test_data(args_src)
    feat_src, label_src = visu_feat(ext, srcloader, os.path.join(args.outf, 'original.pdf'))

    feat_tar, label_tar = visu_feat(ext, teloader, os.path.join(args.outf, args.corruption + '_test_class.pdf'))
    comp_feat(feat_src, label_src, feat_tar, label_tar, os.path.join(args.outf, args.corruption + '_test_marginal.pdf'))

all_err_cls = []

print('Running...')
print('Error (%)\t\ttest\t\tshot')

err_cls = test(teloader, net)[0]
print(('Epoch %d/%d:' %(0, args.nepoch)).ljust(24) +
            '%.2f\t\t' %(err_cls*100))

# -------------------------------

optimizer = optim.SGD(ssh.parameters(), lr=args.lr, momentum=0.9)
optimizer = op_copy(optimizer)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    'min', factor=0.5, patience=10, cooldown=10,
    threshold=0.0001, threshold_mode='rel', min_lr=0.0001, verbose=True)

# ----------- Improved Test-time Training ------------

losses = AverageMeter('Loss', ':.4e')

for epoch in range(1, args.nepoch+1):

    tic = time.time()
    ext.eval()
    mem_label = obtain_shot_label(trloader, ext, classifier, args)
    mem_label = torch.from_numpy(mem_label).cuda()
    ext.train()

    # optimizer = lr_scheduler(optimizer, epoch, 30)
    for batch_idx, (inputs, labels) in enumerate(trloader):

        optimizer.zero_grad()
        classifier_loss = 0
        
        features_test = ext(inputs.cuda())
        outputs_test = classifier(features_test)

        if args.cls_par > 0:
            pred = mem_label[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        classifier_loss.backward()
        optimizer.step()
        losses.update(classifier_loss.item(), labels.size(0))

    err_cls = test(teloader, net)[0]
    all_err_cls.append(err_cls)

    toc = time.time()
    print(('Epoch %d/%d (%.0fs):' %(epoch, args.nepoch, toc-tic)).ljust(24) +
                    '%.2f\t\t' %(err_cls*100) +
                    '{loss.val:.4f}'.format(loss=losses))

    # termination and save
    if epoch > (args.stopepoch + 1) and all_err_cls[-args.stopepoch] < min(all_err_cls[-args.stopepoch+1:]):
        print("Termination: {:.2f}".format(all_err_cls[-args.stopepoch]*100))
        # state = {'net': net.state_dict(), 'head': head.state_dict()}
        # save_file = os.path.join(args.outf, args.corruption + '_' +  args.method + '.pth')
        # torch.save(state, save_file)
        # print('Save model to', save_file)
        break

    # lr decay
    scheduler.step(err_cls)

# -------------------------------

if args.method == 'shot':
    prefix = os.path.join(args.outf, args.corruption + '_shot')
else:
    raise NotImplementedError

if args.tsne:
    feat_tar, label_tar = visu_feat(ext, teloader, prefix+'_class.pdf')
    comp_feat(feat_src, label_src, feat_tar, label_tar, prefix+'_marginal.pdf')

# -------------------------------

# df = pd.DataFrame([all_err_cls]).T
# df.to_csv(prefix, index=False, float_format='%.4f', header=False)
