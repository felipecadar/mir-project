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
from vae_model import Autoencoder

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import kornia

import librosa
import soundfile as sf
import gc


if __name__ == "__main__":

    testset = SongsDataset(simplified=False, train=False, baby=False)
    testloader = DataLoader(testset, batch_size=1)

    EFFECTS = testset.EFFECTS

    clean_sample, effect_samples = testset.getEvalSample()

    model = Autoencoder(H=clean_sample.shape[-2], W=clean_sample.shape[-1])
    criterion = kornia.losses.SSIMLoss(11)  # nn.MSELoss()

    model.load_state_dict(torch.load(sys.argv[1]))

    clean_reconstructed, clean_lv = model(clean_sample, latent=True)
    
    effects_lv = {}
    for effect_name in EFFECTS:
        effects_lv[effect_name] = model(effect_samples[effect_name])[1] - clean_lv #only distortion latent vector

    add_results = [ [] for _ in EFFECTS ]
    remove_results = [ [] for _ in EFFECTS ]
    y_pos = [ i for i in range(len(EFFECTS)) ]

    i = -1
    for data in tqdm(testloader):
        gc.collect()
        i += 1 

        x, y, label_idx = data

        label_idx = label_idx[0]
        # x = x.unsqueeze(0)
        # y = y.unsqueeze(0)

        x_reconstructed, x_latent = model(x, latent=True)
        y_reconstructed, y_latent = model(y, latent=True)
        
        # # Remove batch dim
        # x_reconstructed, x_latent = x_reconstructed.squeeze(0), x_latent.squeeze(0)
        # y_reconstructed, y_latent = y_reconstructed.squeeze(0), y_latent.squeeze(0)

        # # apply distortion
        x_latent_effect = x_latent + effects_lv[EFFECTS[label_idx]]
        x_reconstructed_effect = model.decode(x_latent_effect)
        add_result = criterion(x_reconstructed_effect, y).item()

        # remove distortion
        y_latent_clean = y_latent - effects_lv[EFFECTS[label_idx]]
        y_reconstructed_clean = model.decode(y_latent_clean)
        remove_result = criterion(y_reconstructed_clean, x).item()

        # np_x = x.detach().numpy()[0,0,:,:]
        # np_y = y.detach().numpy()[0,0,:,:]
        # np_x_reconstructed = x_reconstructed.detach().numpy()[0,0,:,:]
        # np_y_reconstructed = y_reconstructed.detach().numpy()[0,0,:,:]
        # np_x_reconstructed_effect = x_reconstructed_effect.detach().numpy()[0,0,:,:]
        # np_y_reconstructed_clean = y_reconstructed_clean.detach().numpy()[0,0,:,:]

        # wave_x_original = denorm_sound(np_x)
        # wave_y_original = denorm_sound(np_y)

        # wave_x_reconstructed = denorm_sound(np_x_reconstructed)
        # wave_y_reconstructed = denorm_sound(np_y_reconstructed)


        # wave_x_reconstructed_effect = denorm_sound(np_x_reconstructed_effect)
        # wave_y_reconstructed_clean = denorm_sound(np_y_reconstructed_clean)

        # fig, axes = plt.subplots(3, 2, figsize=(15,15))

        # axes[0,0].imshow(np_x, interpolation='nearest', aspect='auto')
        # axes[0,0].set_title("x")
        # axes[0,1].imshow(np_y, interpolation='nearest', aspect='auto')
        # axes[0,1].set_title("y")
        # axes[1,0].imshow(np_x_reconstructed, interpolation='nearest', aspect='auto')
        # axes[1,0].set_title("x_reconstructed")
        # axes[1,1].imshow(np_y_reconstructed, interpolation='nearest', aspect='auto')
        # axes[1,1].set_title("y_reconstructed")
        # axes[2,0].imshow(np_x_reconstructed_effect, interpolation='nearest', aspect='auto')
        # axes[2,0].set_title("x_reconstructed_effect")
        # axes[2,1].imshow(np_y_reconstructed_clean, interpolation='nearest', aspect='auto')
        # axes[2,1].set_title("y_reconstructed_clean")

        # plt.savefig("audios/{}_specs.png".format(i))
        # plt.close()

        # sf.write("audios/{}_wave_x_original.wav".format(i), wave_x_original, 22050, subtype='PCM_24')
        # sf.write("audios/{}_wave_y_original.wav".format(i), wave_y_original, 22050, subtype='PCM_24')
        # sf.write("audios/{}_wave_x_reconstructed.wav".format(i), wave_x_reconstructed, 22050, subtype='PCM_24')
        # sf.write("audios/{}_wave_y_reconstructed.wav".format(i), wave_y_reconstructed, 22050, subtype='PCM_24')
        # sf.write("audios/{}_wave_x_reconstructed_effect.wav".format(i), wave_x_reconstructed_effect, 22050, subtype='PCM_24')
        # sf.write("audios/{}_wave_y_reconstructed_clean.wav".format(i), wave_y_reconstructed_clean, 22050, subtype='PCM_24')

        add_results[label_idx].append(add_result)
        remove_results[label_idx].append(remove_result)

        # import pdb; pdb.set_trace()

    mean_add = []
    std_add = []

    mean_remove = []
    std_remove = []

    for i in y_pos:
        
        mean = np.mean(add_results[i])
        std = np.std(add_results[i])
    
        mean_add.append(mean)
        std_add.append(std)

        mean = np.mean(remove_results[i])
        std = np.std(remove_results[i])

        mean_remove.append(mean)
        std_remove.append(std)

    mean_add = np.nan_to_num(mean_add)
    std_add = np.nan_to_num(std_add)
    mean_remove = np.nan_to_num(mean_remove)
    std_remove = np.nan_to_num(std_remove)

    fig, ax = plt.subplots()

    hbars = ax.barh(y_pos, mean_add, xerr=std_add, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(EFFECTS)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('SSIM')
    ax.set_title('Add effect performance')

    # Label with specially formatted floats
    plt.savefig("Add.png")
    plt.show()

    fig, ax = plt.subplots()

    hbars = ax.barh(y_pos, mean_remove, xerr=std_remove, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(EFFECTS)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('SSIM')
    ax.set_title('Remove effect performance')

    # Label with specially formatted floats

    plt.savefig("Remove.png")
    plt.show()




