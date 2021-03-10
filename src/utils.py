from IPython.display import Audio 
from IPython.core.display import display

from matplotlib import pyplot as plt

import librosa as lr
import librosa.display

import os
from glob import glob

DATASET_PATH = "/homeLocal/IDMT-SMT-AUDIO-EFFECTS/"
GUITAR_POLY = "Gitarre polyphon"
GUITAR_MONO = "Gitarre monophon"
EFFECTS = ["Chorus", "Distortion", "EQ", "FeedbackDelay", "Flanger"]
CLEAN = "NoFX"

def play(y, sr):
    lr.display.waveplot(y, sr)
    display(Audio(y, rate=sr))
    
## Build Dataset
def makeDataset():
    DATABASE = {}

    for clean_fname in sorted(glob(os.path.join(DATASET_PATH, GUITAR_POLY, "Samples", CLEAN, "*"))):
        base_name = "-".join(os.path.basename(clean_fname).split("-")[:2])

        effects_database = {}
        for effect_name in EFFECTS:
            corresp_songs = glob(os.path.join(DATASET_PATH, GUITAR_POLY, "Samples", effect_name, "%s*" % base_name))
            effects_database[effect_name] = corresp_songs

        DATABASE[clean_fname] = effects_database
    return DATABASE

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise