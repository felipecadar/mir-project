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

from torch.autograd import Variable

from loader import SongsDataset
# from model import Discriminator, Generator
from simple_model import Discriminator, Generator

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


def parseArgs():
    parser = argparse.ArgumentParser("Train the net \o/")
    parser.add_argument("-s", "--save-path", type=str, default="",
                        help="Save path, default uses tensorboard logdir")
    parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("-me", "--max-epochs", type=int,
                        default=100, help="Max Iterations")
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

    batch_size = 1
    loader = DataLoader(dataset, batch_size=batch_size)
    
    n_classes = 5
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    adversarial_loss = torch.nn.MSELoss().to(device)

    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    sample_idx = 0  # index from batch to display
    for epoch in tqdm(range(args.max_epochs), position=1, desc="Epochs"):
        for idx, item in tqdm(enumerate(loader), total=len(loader), desc="Batch"):
            
            # Global index for logging
            global_idx = idx + (epoch * len(loader))

            # unpack dataset
            # tqdm.write("Unpack")
            x,y,labels = item

            # # Move to decive (probably 'cuda:0')
            # x = x.to(device)
            # y = y.to(device)
            # label = label.to(device)

            # Configure input
            # tqdm.write("Sending 3 to device")
            clean_songs = Variable(x.type(FloatTensor)).to(device)
            effect_songs = Variable(y.type(FloatTensor)).to(device)
            labels = Variable(labels.type(LongTensor)).to(device)

            # print(clean_songs.shape)
            # print(effect_songs.shape)
            # print(labels.shape)

            # Adversarial ground truths
            # tqdm.write("create valid/fake")
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).unsqueeze(1).to(device)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).unsqueeze(1).to(device)

            # -----------------
            #  Train Generator
            # -----------------
    
            # tqdm.write("G zero grad")
            optimizer_G.zero_grad()

            # Sample labels as generator input
            # tqdm.write("gen labels")
            gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size))).unsqueeze(1).to(device)
            # print(gen_labels.shape)

            # Generate a batch of songs
            # tqdm.write("gen songs")
            gen_songs = generator(clean_songs, gen_labels)

            # tqdm.write("g loss")
            validity = discriminator(gen_songs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            
            # tqdm.write("g loss back")
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # tqdm.write("d zero grad")
            optimizer_D.zero_grad()

            # Loss for real songs
            # tqdm.write("d loss real")
            validity_real = discriminator(effect_songs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake songs
            # tqdm.write("d loss fake")
            validity_fake = discriminator(gen_songs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # tqdm.write("d loss med")
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # tqdm.write("d loss back")
            d_loss.backward()
            # tqdm.write("d loss step")
            optimizer_D.step()

            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.max_epochs, idx, len(loader), d_loss.item(), g_loss.item())
                )

            # Logging...
            writer.add_scalar("discriminator/loss", d_loss.item(), global_idx)
            writer.add_scalar("generator/loss",  g_loss.item(), global_idx)

            # Audio Logging
            if idx % 10 == 0:
                writer.add_audio("generator/gen-songs",
                                 gen_songs[sample_idx, :, :].view(-1), global_idx, sample_rate=44100)

            # Figure Logging
            if idx % 10 == 0:
                fig = plt.figure()
                curve = gen_songs[sample_idx, :, :].view(-1).cpu().detach().numpy()
                plt.plot(curve)
                writer.add_figure(
                    "generator/gen-songs-fig", fig, global_idx)
                plt.clf()

            #     fig = plt.figure()
            #     curve = x[sample_idx, :, :].view(-1).cpu().detach().numpy()
            #     plt.plot(curve)
            #     writer.add_figure("distortion/plot-original/", fig, global_idx)
            #     plt.clf()

            #     fig = plt.figure()
            #     curve = y[sample_idx, :, :].view(-1).cpu().detach().numpy()
            #     plt.plot(curve)
            #     writer.add_figure("distortion/plot-target/", fig, global_idx)
            #     plt.clf()

    writer.close()
