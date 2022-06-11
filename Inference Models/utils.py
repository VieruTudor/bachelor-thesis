import os

import librosa
import numpy as np

from constants import *


def get_ravdess_label(file):
    parts = file.split('-')
    return ravdess_to_emotions[parts[2]]


def get_tess_label(file):
    parts = file.split('_')
    return tess_to_emotions[parts[2][:-4]]


def get_savee_label(file):
    file = file[:-4]
    if len(file) == 3:
        return savee_to_emotions[file[0:1]]
    else:
        return savee_to_emotions[file[0:2]]
