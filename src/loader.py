 
from torch.utils.data import Dataset, DataLoader
import torch.tensor
import torch
from glob import glob
import os
import numpy as np
import random
import librosa as lr
from tqdm import tqdm


class SongsDataset(Dataset):
    def __init__(self, root_dir="/homeLocal/IDMT-SMT-AUDIO-EFFECTS/", mono=False, train=True, split=0.9, simplified=False, baby=True):

        self.DATASET_PATH = root_dir
        self.baby = baby
        self.train = train
        self.simplified = simplified
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
            
            if self.simplified:
                self.indexes[i] = clean_fname
                i += 1
            
            for effect_idx, effect_name in enumerate(self.EFFECTS):
                for alternative in range(len(self.DATABASE[clean_fname][effect_name])):
                    if self.simplified:
                        self.indexes[i] = self.DATABASE[clean_fname][effect_name][alternative]
                    else:
                        self.indexes[i] = (clean_fname, self.DATABASE[clean_fname][effect_name][alternative], effect_idx)
    
                    i += 1

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        if self.baby:
            idx = 0
        
        if not self.simplified:
            clean_fname, effect_fname, effect_idx = self.indexes[idx]
            x, sr = lr.load(clean_fname)
            y, _ = lr.load(effect_fname)

            # x = np.abs(lr.stft(x))
            # y = np.abs(lr.stft(y))

            x = torch.tensor([x], requires_grad=True)
            y = torch.tensor([y], requires_grad=True)
            context = torch.LongTensor([effect_idx])

            return x,y,context

        if self.simplified:
            fname = self.indexes[idx]
            x, sr = lr.load(fname)
            
            # x = np.abs(lr.stft(x))
            x = torch.tensor([x], requires_grad=True)
            return x


    def getNp(self, idx):
        clean_fname = self.indexes[idx]
        x, sr = lr.load(clean_fname)
        return x

    def getMeanStd(self, recalc=False):

        if recalc == False:
            # Mean: 0.08186184
            # Std: 0.56715244
            return [0.08186184], [0.56715244]
        else:
            subset = list(range(len(self.indexes)))
            random.shuffle(subset)
            subset = subset[:500]

            means = []
            stds = []
            for idx in tqdm(subset):
                x = self.getNp(idx)

                x_stft = np.abs(lr.stft(x))

                means.append(np.mean(x_stft))
                stds.append(np.std(x_stft))

            mean = np.mean(means)
            std = np.mean(stds)

            return mean, std


if __name__ == "__main__":

    dataset = SongsDataset(simplified=True)
    print(dataset[0].shape)
    # print(dataset.getMeanStd(recalc=True))