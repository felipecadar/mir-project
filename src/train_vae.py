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


def train(model, dataloader, dataset, device, optimizer, criterion, writer, epoch):
    model.train()
    running_loss = 0.0
    counter = 0

    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size), position=0, desc="Train"):
        counter += 1

        # data = norm(data)
        # maxx = torch.max(data)
        data = data.to(device)

        # import pdb; pdb.set_trace()

        optimizer.zero_grad()

        # reconstruction = model(data)
        reconstruction, mu, logvar = model(data)

        # loss = criterion(reconstruction, data)
        bce_loss = criterion(reconstruction, data)
        kld_loss = KLDloss(mu, logvar)
        loss = final_loss(bce_loss, kld_loss)

        loss.backward()

        running_loss += loss.item()

        global_idx = (int(len(dataset)/dataloader.batch_size) * epoch) + i
        writer.add_scalar("train/loss", loss.item(), global_idx)

        writer.add_scalar("train/kld", kld_loss.item(), global_idx)
        writer.add_scalar("train/ssim", bce_loss.item(), global_idx)

        if i % 10 == 0:

            np_original = data[0, 0, :, :].detach().cpu().numpy()
            np_reconstruction = reconstruction[0,0, :, :].detach().cpu().numpy()

            # wave_original = denorm_sound(np_original)
            # wave_reconstruction = denorm_sound(np_reconstruction)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fig1 = plt.figure()
                imgplot = plt.imshow(
                    np_reconstruction, interpolation='nearest', aspect='auto')
                plt.colorbar()

                fig2 = plt.figure()
                imgplot = plt.imshow(
                    np_original, interpolation='nearest', aspect='auto')
                plt.colorbar()

                writer.add_figure("train/spec-reconstructed", fig1, global_idx)
                writer.add_figure("train/spec-original", fig2, global_idx)

                # writer.add_audio("train/audio-original", wave_original, global_idx)
                # writer.add_audio("train/audio-reconstructed", wave_reconstruction, global_idx)

        optimizer.step()

    train_loss = running_loss / counter

    return train_loss


def validate(model, dataloader, dataset, device, criterion, writer, epoch):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size), position=0, desc="Validation"):
            counter += 1

            data = data.to(device)

            reconstruction, mu, logvar = model(data)
            # reconstruction = model(data)

            # loss = criterion(reconstruction, data)
            bce_loss = criterion(reconstruction, data)
            kld_loss = KLDloss(mu, logvar)
            loss = final_loss(bce_loss, kld_loss)

            running_loss += loss.item()

            global_idx = (int(len(dataset)/dataloader.batch_size) * epoch) + i
            writer.add_scalar("validation/loss", loss.item(), global_idx)

            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                # denorm
                np_data = data[0, 0, :, :].detach().cpu().numpy()
                np_reconstruction = reconstruction[0, 0, :, :].detach().cpu().numpy()
                
                wave_original = denorm_sound(np_data)
                wave_reconstruction = denorm_sound(np_reconstruction)

                writer.add_audio("validation/audio-original",
                                 wave_original, global_idx)
                writer.add_audio("validation/audio-reconstrucation",
                                 wave_reconstruction, global_idx)

    val_loss = running_loss / counter
    return val_loss


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
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = SongsDataset(simplified=True, train=False, baby=args.baby)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    sample = trainset[0]

    # model = ConvVAE().to(device)
    model = Autoencoder(H=sample.shape[-2], W=sample.shape[-1]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    # criterion = nn.MSELoss(reduction='sum')
    criterion = kornia.losses.SSIMLoss(11)  # nn.MSELoss()

    # criterion = nn.BCELoss(reduction='sum')

    running_loss = 0.0
    counter = 0

    train_loss = []
    valid_loss = []

    try:

        for epoch in tqdm(range(args.max_epochs), position=1, desc="Epochs"):
            train_epoch_loss = train(
                model, trainloader, trainset, device, optimizer, criterion, writer, epoch
            )
            valid_epoch_loss = validate(
                model, testloader, testset, device, criterion, writer, epoch
            )
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)

            writer.add_scalar("train/epoch-loss", train_epoch_loss, epoch)
            writer.add_scalar("validation/epoch-loss", valid_epoch_loss, epoch)

            tqdm.write(f"Train Loss: {train_epoch_loss:.4f}")
            tqdm.write(f"Val Loss: {valid_epoch_loss:.4f}")
            if epoch % 50 == 0:
                torch.save(model.state_dict(), os.path.join(logdir, "{}_weights.torch".format(epoch)))

    except KeyboardInterrupt:
        pass
    
    torch.save(model.state_dict(), os.path.join(logdir, "weights.torch"))

    writer.close()
