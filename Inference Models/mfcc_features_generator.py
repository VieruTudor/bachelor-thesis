import os
import librosa
import joblib
import numpy as np

from constants import *
from utils import *

SAMPLE_RATE = 16000

BALANCE_THRESHOLD = 592

balanced_dict = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0
}

def process_ravdess_files():
    mfccs_list = []

    for subdir, dirs, files in os.walk(RAVDESS_PATH):
        for file in files:
            audio_path = os.path.join(subdir, file)
            audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)

            audio = librosa.effects.trim(audio)[0]

            mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
            spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=40).T, axis=0)

            label = get_ravdess_label(file)
            if balanced_dict[label] < BALANCE_THRESHOLD:
                mfccs_list.append(((mfcc, spectrogram), label))
                balanced_dict[label] += 1

    return mfccs_list


def process_tess_files():
    mfccs_list = []

    for subdir, dirs, files in os.walk(TESS_PATH):
        for file in files:
            audio_path = os.path.join(subdir, file)
            audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)

            audio = librosa.effects.trim(audio)[0]

            mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
            spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=40).T, axis=0)

            label = get_tess_label(file)

            if balanced_dict[label] < BALANCE_THRESHOLD:
                mfccs_list.append(((mfcc, spectrogram), label))
                balanced_dict[label] += 1
    return mfccs_list


def get_savee_test_data():
    mfccs_list = []
    for subdir, dirs, files in os.walk(SAVEE_PATH):
        for file in files:
            audio_path = os.path.join(subdir, file)
            audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)

            audio = librosa.effects.trim(audio)[0]

            mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
            spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=40).T, axis=0)

            label = get_savee_label(file)

            mfccs_list.append(((mfcc, spectrogram), label))

    x_test, y_test = zip(*mfccs_list)
    x_test, y_test = np.asarray(x_test), np.asarray(y_test)

    joblib.dump(x_test, X_TEST_DATA)
    joblib.dump(y_test, Y_TEST_DATA)


def process_audio_files_to_dataset():
    results = []

    results += process_ravdess_files()
    results += process_tess_files()

    x, y = zip(*results)

    x, y = np.asarray(x), np.asarray(y)

    joblib.dump(x, X_TRAIN_DATA)
    joblib.dump(y, Y_TRAIN_DATA)


process_audio_files_to_dataset()
get_savee_test_data()