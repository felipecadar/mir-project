 
from torch.utils.data import Dataset, DataLoader
import torch.tensor
import torch
from glob import glob
import os
import numpy as np
import random
from tqdm import tqdm

import librosa
from librosa.feature import melspectrogram
from librosa.feature.inverse import mel_to_audio

def norm_sound(x,sr=22050):
  x = melspectrogram(x,sr=sr)
  x = librosa.power_to_db(x, ref=np.max)
  x = (x+40.)/300.  # + 100 / 200
  return x

def denorm_sound(x,sr=22050):
  x = x*300. - 40.
  x = librosa.db_to_power(x)
  x = mel_to_audio(x)
  return x 


class SongsDataset(Dataset):
    def __init__(self, root_dir="/homeLocal/IDMT-SMT-AUDIO-EFFECTS/", mono=False, train=True, split=0.9, simplified=False, baby=False):

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
        self.DATABASE = {}
        self.mono = mono
        self.makeDataset()
        self.makeIndex()

        self.cache = {}

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


        if self.simplified:
            for clean_fname in CLEAN_DATASET:
                
                self.indexes[i] = clean_fname
                i += 1
                
                for effect_idx, effect_name in enumerate(self.EFFECTS):
                    for alternative in range(len(self.DATABASE[clean_fname][effect_name])):
                        self.indexes[i] = self.DATABASE[clean_fname][effect_name][alternative]
                        i += 1
        else:
            for clean_fname in CLEAN_DATASET:
                for effect_idx, effect_name in enumerate(self.EFFECTS):
                    self.indexes[i] = (clean_fname, self.DATABASE[clean_fname][effect_name][0], effect_idx)
                    i += 1

    def __len__(self):
        return len(self.indexes)
    
    def grab(self, fname):
        # return librosa.load(fname)

        if not (fname in self.cache):
            self.cache[fname] = librosa.load(fname)
            
        return self.cache[fname]

    def __getitem__(self, idx):
        if self.baby:
            idx = idx % 4
        
        if not self.simplified:
            clean_fname, effect_fname, effect_idx = self.indexes[idx]
            x, sr = self.grab(clean_fname)
            y, _ = self.grab(effect_fname)

            x = norm_sound(x)
            y = norm_sound(y)

            x = torch.tensor([x], requires_grad=True)
            y = torch.tensor([y], requires_grad=True)

            # context = torch.LongTensor([effect_idx])

            return x,y,effect_idx

        if self.simplified:
            fname = self.indexes[idx]
            x, sr = self.grab(fname)


            x = norm_sound(x)
            x = torch.tensor([x], requires_grad=True)

            # x_stft = np.abs(librosastft(x))
            # x = torch.tensor([x_stft], requires_grad=True)
            return x

    def getEvalSample(self):
        clean_fname = random.choice(list(self.DATABASE.keys()))
        clean_sample = norm_sound(self.grab(clean_fname)[0])
        clean_sample = torch.tensor([[clean_sample]])

        
        effect_vectors = {effect_name:torch.tensor([[norm_sound(self.grab(self.DATABASE[clean_fname][effect_name][0])[0])]]) for effect_name in self.EFFECTS}

        return clean_sample, effect_vectors

    def getSampleBatch(self):
        clean_fname = random.choice(list(self.DATABASE.keys()))
        clean_sample = norm_sound(self.grab(clean_fname)[0])

        clean_sample = torch.tensor([[clean_sample]])
        effect_vectors = [torch.tensor([[norm_sound(self.grab(self.DATABASE[clean_fname][effect_name][0])[0])]]) for effect_name in self.EFFECTS]

        x = torch.cat(effect_vectors, 0)
        x = torch.cat([x, clean_sample], 0)

        # distortion_sample = torch.tensor([[norm_sound(self.grab(self.DATABASE[clean_fname][self.DISTORTION][0])[0])]])
        # x = torch.cat([clean_sample, distortion_sample], 0)

        return x


    def getNp(self, idx):
        clean_fname = self.indexes[idx]
        x, sr = self.grab(clean_fname)
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

                x_stft = np.abs(librosastft(x))

                means.append(np.mean(x_stft))
                stds.append(np.std(x_stft))

            mean = np.mean(means)
            std = np.mean(stds)

            return mean, std


if __name__ == "__main__":

    dataset = SongsDataset(simplified=False)

    batch = dataset.getSampleBatch()
    print(batch.shape)

    # loader = DataLoader(dataset, batch_size=1)
    # for data in loader:
    #     x, y, i =  data   
    
    #     print(i)