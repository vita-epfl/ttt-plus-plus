"""System module."""
import copy
import torch
from torch import optim
from torch import nn
import tent
from shot_utils import Entropy
from shot_utils import obtain_shot_label
from discrepancy import covariance, coral, linear_mmd


def train(net, x, y, a):
    # train model on source dataset
    criterion_main = nn.BCEWithLogitsLoss()
    criterion_ssl = nn.BCEWithLogitsLoss()
    lr = 1e-2
    niter = 1000
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for i in range(niter):
        om, os, _ = net(x)

        loss_main = criterion_main(om, y.unsqueeze(1))

        loss_ssl = criterion_ssl(os, a.unsqueeze(1))

        loss = loss_main + loss_ssl * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(net, x, y, a):
    # forward pass
    net.eval()
    with torch.no_grad():
        om, os, _ = net(x)
    acc_m = (om.squeeze().gt(0.0) == y).sum() / y.size(0)
    acc_s = (os.squeeze().gt(0.0) == a).sum() / a.size(0)
    return acc_m, acc_s


def summarize(z):
    # feature summarization
    mu = z.mean(axis=0)
    sigma = covariance(z)
    return mu, sigma


def ttt_adapt(net, x, y, a, niter=50000, mu=None, sigma=None, coef=[1.0, 0.1, 1.0]):
    # adapt model at test time
    criterion_ssl = nn.BCEWithLogitsLoss()
    lr = 1e-3
    optimizer = optim.Adam(net.parameters(), lr=lr)

    acc_main_best = 0.0
    acc_main_ema = 0.0
    acc_ssl_ema = 1.0

    for i in range(niter):
        _, os, z = net(x)

        loss_ssl = criterion_ssl(os, a.unsqueeze(1))

        # feature alignment
        loss_mean = linear_mmd(z.mean(axis=0), mu)
        loss_coral = coral(covariance(z), sigma)

        loss = loss_ssl * coef[0] + loss_mean * coef[1] + loss_coral * coef[2]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save best result
        if i % 5 == 0:
            acc = test(net, x, y, a)
            acc_main_ema = acc_main_ema * 0.9 + acc[0] * 0.1
            acc_ssl_ema = acc_ssl_ema * 0.9 + acc[1] * 0.1
            if acc_main_best <= acc[0]:
                acc_main_best = acc[0]
                net_bkp = copy.deepcopy(net.state_dict())

            # termination condition
            if acc[0] > 0.999 or (acc[0] < acc_main_ema and acc[1] > acc_ssl_ema):
                break

    return acc_main_best, net_bkp

def tent_adapt(net, x, y, a, niter=50000):
    # adapt model at test time
    lr = 1e-3
    optimizer = optim.Adam(net.parameters(), lr=lr)
    tent_model = tent.Tent(net, optimizer)
    acc_best = 0
    for i in range(niter):

        outputs = tent_model(x)
        # save best result
        if i % 5 == 0:
            acc, _ = test(tent_model.model, x, y, a)
            if acc_best <= acc:
                acc_best = acc
                net_bkp = copy.deepcopy(net.state_dict())

    return acc, net_bkp

def shot_adapt(net, x, y, a, niter=50000):
    ext = net.encoder
    classifier = net.cls
    # adapt model at test time
    lr = 1e-3
    cls_par = 1e-3
    optimizer = optim.Adam(net.parameters(), lr=lr)

    acc_best = 0

    for i in range(niter):
        ext.eval()
        mem_label = obtain_shot_label(x, ext, classifier)
        mem_label = torch.from_numpy(mem_label)
        ext.train()
        optimizer.zero_grad()
        classifier_loss = 0

        features_test = ext(x)
        outputs_test = classifier(features_test)

        classifier_loss = cls_par * nn.BCEWithLogitsLoss()(outputs_test, mem_label.unsqueeze(1).float())

        sigmoid_out = torch.sigmoid(outputs_test)
        entropy_loss = torch.mean(Entropy(sigmoid_out))
        msigmoid = sigmoid_out.mean(dim=0)
        entropy_loss -= torch.sum(-msigmoid * torch.log(msigmoid + 1e-5))

        im_loss = entropy_loss
        classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        # save best result
        if i % 5 == 0:
            acc, _ = test(net, x, y, a)
            if acc_best <= acc:
                acc_best = acc
                net_bkp = copy.deepcopy(net.state_dict())

    return acc_best, net_bkp

