import argparse
import errno
import glob
import os
import sys

from tqdm import tqdm
import numpy as np

import torch
from torch import load, nn, optim, save
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from loader import SongsDataset
from model import EffectModel

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


def parseArgs():
    parser = argparse.ArgumentParser("Train the net \o/")
    parser.add_argument("-s", "--save-path", type=str, default="",
                        help="Save path, default uses tensorboard logdir")
    parser.add_argument("-lr", "--learning-rate", type=float,
                        default=1e-8, help="Learning rate")
    parser.add_argument("-mi", "--max-iter", type=int,
                        default=1000, help="Max Iterations")
    parser.add_argument('--seed', type=int, default=789,
                        help='random seed (default: 789)')
    return parser.parse_args()


def printArg(args):
    print("-" * 10, flush=True)
    for key, item in args.__dict__.items():
        print("{:>13} -> {}".format(key, item), flush=True)
    print("-" * 10, flush=True)


if __name__ == "__main__":

    args = parseArgs()

    torch.manual_seed(args.seed)

    printArg(args)

    writer = SummaryWriter(args.save_path, flush_secs=10)
    logdir = writer.logdir
    print("Logdir:", logdir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Using Device:", device, flush=True)

    dataset = SongsDataset()
    loader = DataLoader(dataset, batch_size=1)

    for idx, item in enumerate(dataset):
        x, x_sr, y, y_sr = item

        x.to(device)
        y.to(device)

        model = EffectModel(3, 10001).to(device)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.SGD(trainable_params, lr=args.learning_rate, momentum=0.9)

        MSEloss = nn.MSELoss(reduction="sum").to(device)
        # MSEloss = nn.L1Loss().to(device)

        writer.add_audio("{:02d}/original".format(idx), x, 0, sample_rate=x_sr)
        writer.add_audio("{:02d}/effect".format(idx), y, 0, sample_rate=y_sr)

        x = x.unsqueeze(0).unsqueeze(0).to(device)
        y = y.unsqueeze(0).unsqueeze(0).to(device)

        for i in tqdm(range(args.max_iter)):
            optimizer.zero_grad()

            new_x = model(x)
            # print(new_x.shape)

            loss = MSEloss(new_x, y)
            loss.backward()

            optimizer.step()

            writer.add_scalar("{:02d}/loss".format(idx), loss.item(), i)

            if i % 10 == 0:
                # for param_idx, param in enumerate(model.parameters()):
                #     if len(param.shape) > 1:
                #         fig = plt.figure()
                #         curve = param.view(-1).cpu().detach().numpy()
                #         plt.plot(curve)
                #         writer.add_figure("{:02d}-song/{:02d}-filter/".format(idx, param_idx), fig, i)
                #         plt.clf()

                writer.add_audio("{:02d}/processed".format(idx),
                                new_x.squeeze(0).squeeze(0), i, sample_rate=x_sr)

        break

    writer.close()
