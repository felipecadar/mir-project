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

from loader import SongsDataset
from vae_model import ConvVAE, final_loss

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import librosa
import librosa.display

def parseArgs():
    parser = argparse.ArgumentParser("Train the net \o/")
    parser.add_argument("-s", "--save-path", type=str, default="",help="Save path, default uses tensorboard logdir")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("-me", "--max-epochs", type=int,default=100, help="Max Iterations")
    parser.add_argument("-bs", "--batch-size", type=int,default=4, help="Batch Size")
    parser.add_argument('--seed', type=int, default=789,help='random seed (default: 789)')
    return parser.parse_args()


def printArg(args):
    print("-" * 10, flush=True)
    for key, item in args.__dict__.items():
        print("{:>13} -> {}".format(key, item), flush=True)
    print("-" * 10, flush=True)

def norm(x):

    mean=0.14439507
    std=0.89928925
    return (x - mean)/std

def denorm(x):
    mean=0.14439507
    std=0.89928925
    return (x*std) + mean

def train(model, dataloader, dataset, device, optimizer, criterion, writer, epoch):
    model.train()
    running_loss = 0.0
    counter = 0
    
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size), position=0, desc="Train"):
        counter += 1

        data = norm(data)
        data = data.to(device)

        optimizer.zero_grad()

        reconstruction, mu, logvar = model(data)

        # loss = criterion(reconstruction, data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)

        loss.backward()

        running_loss += loss.item()

        global_idx = (int(len(dataset)/dataloader.batch_size) * epoch) + i
        writer.add_scalar("train/loss", loss.item(), global_idx)

        if i % 10 == 0:
            
            np_reconstruction = denorm(reconstruction[0, 0, :, :].detach().cpu().numpy())
            np_original  = denorm(data[0, 0, :, :].detach().cpu().numpy())

            audio_reconstruction = librosa.istft(np_reconstruction)
            audio_original = librosa.istft(np_original)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                fig, ax = plt.subplots()
                img = librosa.display.specshow(librosa.amplitude_to_db(np_reconstruction ,ref=np.max), y_axis='log', x_axis='time', ax=ax)
                ax.set_title('Reconstruction Power spectrogram')
                fig.colorbar(img, ax=ax, format="%+2.0f dB")

                writer.add_figure("train/spec-reconstructed", fig, global_idx)

                fig, ax = plt.subplots()
                img = librosa.display.specshow(librosa.amplitude_to_db(np_original ,ref=np.max), y_axis='log', x_axis='time', ax=ax)
                ax.set_title('Reconstruction Power spectrogram')
                fig.colorbar(img, ax=ax, format="%+2.0f dB")

                writer.add_figure("train/spec-original", fig, global_idx)

                writer.add_audio("train/audio-reconstructed", audio_reconstruction, global_idx, sample_rate=44100)
                writer.add_audio("train/audio-original", audio_original, global_idx, sample_rate=44100)
            
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
            
            # loss = criterion(reconstruction, data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)

            running_loss += loss.item()

            global_idx = (int(len(dataset)/dataloader.batch_size) * epoch) + i
            writer.add_scalar("validation/loss", loss.item(), global_idx)
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction

    val_loss = running_loss / counter
    return val_loss, recon_images

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

    model = ConvVAE().to(device)

    lr = args.lr
    epochs = args.max_epochs
    batch_size = args.batch_size
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')
    # criterion = nn.BCELoss(reduction='sum')
    
    running_loss = 0.0
    counter = 0

    trainset = SongsDataset(simplified=True, train=True, baby=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = SongsDataset(simplified=True, train=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    train_loss = []
    valid_loss = []
    for epoch in tqdm(range(args.max_epochs), position=1, desc="Epochs"):
        train_epoch_loss = train(
            model, trainloader, trainset, device, optimizer, criterion, writer,epoch
        )
        valid_epoch_loss, recon_images = validate(
            model, testloader, testset, device, criterion, writer, epoch
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)

        writer.add_scalar("train/epoch-loss", train_epoch_loss, epoch)
        writer.add_scalar("validation/epoch-loss", valid_epoch_loss, epoch)

        tqdm.write(f"Train Loss: {train_epoch_loss:.4f}")
        tqdm.write(f"Val Loss: {valid_epoch_loss:.4f}")


    writer.close()
