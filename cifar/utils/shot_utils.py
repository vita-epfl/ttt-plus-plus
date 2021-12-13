import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
import numpy as np

def train_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'    
    netC.load_state_dict(torch.load(args.modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    # interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['target_te'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.cuda()
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            # print(log_str+'\n')
            netF.train()
            netB.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC

def obtain_shot_label(loader, ext, task_head, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = ext(inputs)
            outputs = task_head(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    # print(log_str+'\n')
    return pred_label.astype('int')

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer
