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

    trainset = SongsDataset(simplified=True, train=True, baby=False)
    EFFECTS = trainset.EFFECTS

    data = trainset.getSampleBatch()

    model = Autoencoder(H=data.shape[-2], W=data.shape[-1])
    criterion = kornia.losses.SSIMLoss(11)  # nn.MSELoss()

    model.load_state_dict(torch.load(sys.argv[1]))

    reconstruction, rec_latent = model(data, latent=True)    
    
    clean_latent    = rec_latent[0]
    Chorus          = rec_latent[1] - clean_latent
    Distortion      = rec_latent[2] - clean_latent
    EQ              = rec_latent[3] - clean_latent
    FeedbackDelay   = rec_latent[4] - clean_latent
    Flanger         = rec_latent[5] - clean_latent


    clean_latent = clean_latent.unsqueeze(0)
    Chorus = Chorus.unsqueeze(0)
    Distortion = Distortion.unsqueeze(0)
    EQ = EQ.unsqueeze(0)
    FeedbackDelay = FeedbackDelay.unsqueeze(0)
    Flanger = Flanger.unsqueeze(0)


    clean_Flanger = clean_latent + Flanger
    audio_clean_Flanger = denorm_sound(model.decode(clean_Flanger)[0,0,:,:].detach().numpy())    

    print("Writing WAVs")

    sf.write("audios/clean_Flanger.wav", audio_clean_Flanger, 22050, subtype='PCM_24')
    
    sf.write("audios/reconstructed-clean.wav", denorm_sound(reconstruction[0, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/reconstructed-Chorus.wav", denorm_sound(reconstruction[1, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/reconstructed-Distortion.wav", denorm_sound(reconstruction[2, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/reconstructed-EQ.wav", denorm_sound(reconstruction[3, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/reconstructed-FeedbackDelay.wav", denorm_sound(reconstruction[4, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/reconstructed-Flanger.wav", denorm_sound(reconstruction[5, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')

    sf.write("audios/original-clean.wav", denorm_sound(data[0, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/original-Chorus.wav", denorm_sound(data[1, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/original-Distortion.wav", denorm_sound(data[2, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/original-EQ.wav", denorm_sound(data[3, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/original-FeedbackDelay.wav", denorm_sound(data[4, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')
    sf.write("audios/original-Flanger.wav", denorm_sound(data[5, 0, :, :].detach().numpy()), 22050, subtype='PCM_24')



