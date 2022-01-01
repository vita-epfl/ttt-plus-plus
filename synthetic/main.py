import os
import time
import argparse
import copy
import numpy as np
import torch
import algo
import model
import dataset
import visualize
from shot_utils import configure_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rot', default=50, type=float)
    parser.add_argument('--tran', default=1.0, type=float)
    parser.add_argument('--sep', default=-0.5, type=float)
    parser.add_argument('--figdir', default=None, type=str)
    parser.add_argument('--compare_tent_shot', default=True, type=str)
    args = parser.parse_args()
    return args


def run_experiment(rot, tran, sep, figdir=None, seed=2021):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if figdir and (not os.path.exists(figdir)):
        os.mkdir(figdir)

    start = time.time()
    result = np.zeros(7)

    # Dataset
    (input_s, label_s, pretext_s), (input_t, label_t, pretext_t), corr = dataset.sample(rot, tran, sep)

    if figdir:
        visualize.plot_data(input_s, label_s, os.path.join(figdir, 'source_main.png'))
        visualize.plot_data(input_s, pretext_s, os.path.join(figdir, 'source_ssl.png'))
        visualize.plot_data(input_t, label_t, os.path.join(figdir, 'target_main.png'))
        visualize.plot_data(input_t, pretext_t, os.path.join(figdir, 'target_ssl.png'))

    # Model
    net = model.Shallow()

    # Train
    algo.train(net, input_s, label_s, pretext_s)

    # Test
    acc = algo.test(net, input_s, label_s, pretext_s)
    print("Acc   :  Main  |   SSL")
    print("Source: {:.4f} | {:.4f}".format(acc[0], acc[1]))
    acc = algo.test(net, input_t, label_t, pretext_t)
    print("Target: {:.4f} | {:.4f}".format(acc[0], acc[1]))
    print("-"*25)


    if figdir:
        visualize.plot_prediction(input_s, label_s, net, 3, os.path.join(figdir, 'source_test.png'))
        visualize.plot_prediction(input_t, label_t, net, 3, os.path.join(figdir, 'target_test.png'))

    # Param
    net_bkp = copy.deepcopy(net.state_dict())

    # Offline summarization
    with torch.no_grad():
        _, _, z_s = net(input_s)
        mu, sigma = algo.summarize(z_s)
    if args.compare_tent_shot == False:
        # Explore the effects of Task Relation and Domain shift on TTT and TTT++
        result[0:5] = [tran, rot, sep, corr.item(), acc[0].item()]
        # TTT
        net.load_state_dict(net_bkp, strict=True)
        acc, net_state = algo.ttt_adapt(net, input_t, label_t, pretext_t,
                           niter=50000, mu=mu, sigma=sigma,
                           coef=[1.0, 0.0, 0.0])
        print("TTT   Acc: {:.4f}".format(acc.item()))
        result[5] = acc

        if figdir:
            net.load_state_dict(net_state, strict=True)
            visualize.plot_prediction(input_t, label_t, net, 3, os.path.join(figdir, 'target_ttt.png'))

        # TFA
        # net.load_state_dict(net_bkp, strict=True)
        # acc, _ = algo.adapt(
        #              net, input_t, label_t, pretext_t,
        #              mu, sigma, lr=1e-3, coef=[0.0, 0.1, 1.0])
        # print("TFA   Acc: {:.4f}".format(acc.item()))

        # TTT++
        net.load_state_dict(net_bkp, strict=True)

        acc, net_state = algo.ttt_adapt(net, input_t, label_t, pretext_t,
                           niter=50000, mu=mu, sigma=sigma,
                           coef=[1.0, 0.1, 1.0])
        print("TTT++ Acc: {:.4f}".format(acc.item()))
        result[6] = acc

        if figdir:
            net.load_state_dict(net_state, strict=True)
            visualize.plot_prediction(input_t, label_t, net, 3, os.path.join(figdir, 'target_ttt++.png'))

    else:
        # compare TTT++ with shot and tent
        result[0:4] = [tran, rot, sep, acc[0].item()]
        # TTT++
        net.load_state_dict(net_bkp, strict=True)
        configure_model(net)
        acc, net_state = algo.ttt_adapt(net, input_t, label_t, pretext_t,
                                        niter=50000, mu=mu, sigma=sigma,
                                        coef=[1.0, 0.1, 1.0])
        print("TTT++ Acc: {:.4f}".format(acc.item()))
        result[4] = acc

        if figdir:
            net.load_state_dict(net_state, strict=True)
            visualize.plot_prediction(input_t, label_t, net, 3, os.path.join(figdir, 'target_ttt++.png'))

        # Tent
        net.load_state_dict(net_bkp, strict=True)
        configure_model(net)
        acc, net_state = algo.tent_adapt(net, input_t, label_t, pretext_t)
        print("Tent Acc: {:.4f}".format(acc.item()))
        result[5] = acc

        if figdir:
            # net.load_state_dict(net_state, strict=True)
            visualize.plot_prediction(input_t, label_t, net, 3, os.path.join(figdir, 'target_tent.png'))

        # Shot
        net.load_state_dict(net_bkp, strict=True)
        configure_model(net)
        acc, net_state = algo.shot_adapt(net, input_t, label_t, pretext_t)
        print("Shot Acc: {:.4f}".format(acc.item()))
        result[6] = acc

        if figdir:
            # net.load_state_dict(net_state, strict=True)
            visualize.plot_prediction(input_t, label_t, net, 3, os.path.join(figdir, 'target_shot.png'))
    print("Elapsed: {:.2f}s".format(time.time() - start))
    return result


if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(args.rot, args.tran, args.sep, args.figdir)