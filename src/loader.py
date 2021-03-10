 
from torch.utils.data import Dataset, DataLoader
import torch.tensor
import torch
from glob import glob
import os
import numpy as np
import random
import librosa as lr


class SongsDataset(Dataset):
    def __init__(self, root_dir="/homeLocal/IDMT-SMT-AUDIO-EFFECTS/", mono=False):

        self.DATASET_PATH = root_dir
        self.GUITAR_POLY = "Gitarre polyphon"
        self.GUITAR_MONO = "Gitarre monophon"
        self.EFFECTS = ["Chorus", "Distortion", "EQ", "FeedbackDelay", "Flanger"]
        self.CLEAN = "NoFX"
        self.DISTORTION = "Distortion"
        
        self.mono = mono
        self.makeDataset()

    ## Build Dataset
    def makeDataset(self):
        self.DATABASE = {}

        if self.mono:
            for clean_fname in glob(os.path.join(self.DATASET_PATH, self.GUITAR_MONO, "Samples", self.CLEAN, "*")):
                base_name = "-".join(os.path.basename(clean_fname).split("-")[:2])

                effects_database = {}
                for effect_name in self.EFFECTS:
                    corresp_songs = glob(os.path.join(self.DATASET_PATH, self.GUITAR_MONO, "Samples", effect_name, "%s*" % base_name))
                    effects_database[effect_name] = corresp_songs

                self.DATABASE[clean_fname] = effects_database
        else:
            for clean_fname in glob(os.path.join(self.DATASET_PATH, self.GUITAR_POLY, "Samples", self.CLEAN, "*")):
                base_name = "-".join(os.path.basename(clean_fname).split("-")[:2])

                effects_database = {}
                for effect_name in self.EFFECTS:
                    corresp_songs = glob(os.path.join(self.DATASET_PATH, self.GUITAR_POLY, "Samples", effect_name, "%s*" % base_name))
                    effects_database[effect_name] = corresp_songs

                self.DATABASE[clean_fname] = effects_database

        self.indexes = {i:key for i, key in enumerate(self.DATABASE.keys())}

    def __getitem__(self, idx):
        
        x, x_sr = lr.load(self.indexes[idx])
        y, y_sr = lr.load(self.DATABASE[self.indexes[idx]][self.DISTORTION][0])

        if x.shape[0] % 2 == 1:
            x = x[:-1]
            y = y[:-1]

        x = torch.tensor(x, requires_grad=True)
        y = torch.tensor(y, requires_grad=True)

        return x, x_sr, y, y_sr
