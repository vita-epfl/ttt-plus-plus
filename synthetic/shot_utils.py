import torch
from torch import nn
import numpy as np
from scipy.spatial.distance import cdist

def obtain_shot_label(inputs, ext, classifier):
    with torch.no_grad():
        feas = ext(inputs)
        outputs = classifier(feas)
        all_fea = feas.float()
        all_output = outputs.float()
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().numpy()

    K = all_output.size(1)
    aff = all_output.float().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

    return pred_label.astype('int')

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def configure_model(net):
    classifier = net.cls
    # ssh = net.ssh
    for name, param in classifier.named_parameters():
        param.requires_grad = False
    # for name, param in ssh.named_parameters():
        # param.requires_grad = False
