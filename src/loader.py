 
from torch.utils.data import Dataset, DataLoader
import torch.tensor
import torch
from glob import glob
import os
import numpy as np
import random
import librosa as lr


class SongsDataset(Dataset):
    def __init__(self, root_dir="/homeLocal/IDMT-SMT-AUDIO-EFFECTS/", mono=False, train=True, split=0.9):

        self.DATASET_PATH = root_dir
        self.train = train
        self.split = split
        self.GUITAR_POLY = "Gitarre polyphon"
        self.GUITAR_MONO = "Gitarre monophon"
        self.EFFECTS = ["Chorus", "Distortion", "EQ", "FeedbackDelay", "Flanger"]
        self.CLEAN = "NoFX"
        self.DISTORTION = "Distortion"
        
        self.mono = mono
        self.makeDataset()
        self.makeIndex()

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

    def makeIndex(self):
        self.indexes = {}
        i = 0
        CLEAN_DATASET = list(self.DATABASE.keys())
        split_point = int(len(CLEAN_DATASET) * self.split)

        if self.train:
            CLEAN_DATASET = CLEAN_DATASET[:split_point]
        else:
            CLEAN_DATASET = CLEAN_DATASET[split_point:]

        for clean_fname in CLEAN_DATASET:
            for effect_idx, effect_name in enumerate(self.EFFECTS):
                for alternative in range(len(self.DATABASE[clean_fname][effect_name])):
                    self.indexes[i] = (clean_fname, self.DATABASE[clean_fname][effect_name][alternative], effect_idx)
                    i += 1

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        
        clean_fname, effect_fname, effect_idx = self.indexes[idx]
        x, sr = lr.load(clean_fname)
        y, sr = lr.load(effect_fname)

        # if x.shape[0] % 2 == 1:
        #     x = x[:-1]
        #     y = y[:-1]

        x = torch.tensor(x, requires_grad=True).unsqueeze(0)
        y = torch.tensor(y, requires_grad=True).unsqueeze(0)
        context = torch.LongTensor([effect_idx])

        return x,y,context


if __name__ == "__main__":

    dataset = SongsDataset()
    x, y, context = dataset[0]

    print(x.shape)
    print(y.shape)
    print(context.shape)
