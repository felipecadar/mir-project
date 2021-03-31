import argparse
import errno
import glob
import os
import sys
import warnings

from tqdm import tqdm
import numpy as np
import random

import torch
from torch import load, nn, optim, save
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from torch.autograd import Variable

from loader import SongsDataset, denorm_sound
from vae_model import final_loss, Autoencoder, KLDloss

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import kornia

import librosa
import librosa.display


def parseArgs():
    parser = argparse.ArgumentParser("Train the net \o/")
    parser.add_argument("-s", "--save-path", type=str, default="",
                        help="Save path, default uses tensorboard logdir")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("-me", "--max-epochs", type=int,
                        default=10000, help="Max Iterations")
    parser.add_argument("-bs", "--batch-size", type=int,
                        default=16, help="Batch Size")
    parser.add_argument('--seed', type=int, default=789,
                        help='random seed (default: 789)')
    parser.add_argument('--baby', default=False, action='store_true', help="Use baby dataset")
    return parser.parse_args()


def printArg(args):
    print("-" * 10, flush=True)
    for key, item in args.__dict__.items():
        print("{:>13} -> {}".format(key, item), flush=True)
    print("-" * 10, flush=True)


def norm(x):

    mean = 0.14439507
    std = 0.89928925
    return (x - mean)/std


def denorm(x):
    mean = 0.14439507
    std = 0.89928925
    return (x*std) + mean


if __name__ == "__main__":

    args = parseArgs()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    printArg(args)

    writer = SummaryWriter(args.save_path, flush_secs=10)
    logdir = writer.logdir
    print("Logdir:", logdir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("Using Device:", device, flush=True)

    lr = args.lr
    epochs = args.max_epochs
    batch_size = args.batch_size

    trainset = SongsDataset(simplified=True, train=True, baby=args.baby)

    sample = trainset[0]

    # model = ConvVAE().to(device)
    model = Autoencoder(H=sample.shape[-2], W=sample.shape[-1]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    # criterion = nn.MSELoss(reduction='sum')
    criterion = kornia.losses.SSIMLoss(11)  # nn.MSELoss()

    # criterion = nn.BCELoss(reduction='sum')

    running_loss = 0.0
    counter = 0
    alpha = 0.999

    try:
        data = trainset.getSampleBatch()
        data = data.to(device)
        batch_size = data.shape[0]
        for epoch in tqdm(range(args.max_epochs), position=1, desc="Epochs"):

            optimizer.zero_grad()

            # reconstruction = model(data)
            reconstruction, mu, logvar = model(data)

            # loss = criterion(reconstruction, data)
            ssim_loss = criterion(reconstruction, data)
            kld_loss = KLDloss(mu, logvar)
            if epoch < 0:
                loss = ssim_loss
            else:
                loss = (alpha * ssim_loss) + (1-alpha) * kld_loss

            loss.backward()

            writer.add_scalar("train/loss", loss.item(), epoch)
            writer.add_scalar("train/kld", kld_loss.item(), epoch)
            writer.add_scalar("train/ssim", ssim_loss.item(), epoch)

            if epoch % 100 == 0:
                for batch_idx in range(batch_size):
                    np_original = data[batch_idx, 0, :, :].detach().cpu().numpy()
                    np_reconstruction = reconstruction[batch_idx,0, :, :].detach().cpu().numpy()

                    # wave_original = denorm_sound(np_original)
                    # wave_reconstruction = denorm_sound(np_reconstruction)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
                        im1 = axes[0].imshow(np_original, interpolation='nearest', aspect='auto')
                        axes[0].set_title("Original")

                        im2 = axes[1].imshow(np_reconstruction, interpolation='nearest', aspect='auto')
                        axes[1].set_title("Reconstruction")

                        writer.add_figure("train/{}-spec".format(batch_idx), fig, epoch)

                        # writer.add_audio("train/audio-original", wave_original, global_idx)
                        # writer.add_audio("train/audio-reconstructed", wave_reconstruction, global_idx)

            optimizer.step()

            if epoch % 50 == 0:
                torch.save(model.state_dict(), os.path.join(logdir, "{}_weights.torch".format(epoch)))

    except KeyboardInterrupt:
        pass
    
    torch.save(model.state_dict(), os.path.join(logdir, "weights.torch"))

    writer.close()
