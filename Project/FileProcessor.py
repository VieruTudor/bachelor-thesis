import os
import sys

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import gc
from constants import *

root = os.path.dirname(__file__)
dataset_path = f'{root}\\Dataset'


# TODO:  refactor this class
class FileProcessor:
    def convert_audio_to_spectrogram(self, file, save_path):

        samples, sample_rate = librosa.load(file)
        yt, _ = librosa.effects.trim(samples)

        mel_spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=1024, hop_length=100)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(mel_spectrogram, y_axis='mel', fmax=2 ** 14, x_axis='time')

        plt.savefig(save_path)
        plt.title('Mel Spectrogram')

    def process_tess_files(self):
        for folder in os.listdir('Tess_Dataset/Audio'):
            print("starting ", folder)
            tess_emotion = folder.split('_')[-1].strip().lower()
            if tess_emotion == 'fear':
                tess_emotion = 'fearful'
            if tess_emotion in emotions.values():
                for file in os.listdir(f'Tess_Dataset/Audio/{folder}'):
                    save_path = f'Tess_Dataset/Spectrograms/{tess_emotion}/{file[:-4]}.jpg'

                    if f'{file[:-4]}.jpg' not in os.listdir(f'Tess_Dataset/Spectrograms/{tess_emotion}'):
                        self.convert_audio_to_spectrogram(f'Tess_Dataset/Audio/{folder}/{file}', save_path)

    def process_ravdess_files(self):
        for subdir, dir, files in os.walk('Ravdess_Dataset/Spectrograms'):
            for file in files:
                file_name, extension = os.path.splitext(file)
                file_name = os.path.basename(file_name)

                spectrogram_path = f'Ravdess_Dataset/Spectrograms/{file_name}.jpg'

                # convert .wav to spectrogram
                # we only convert files that are starting with 03 - audio only
                if extension == '.wav' and file_name.split('-')[0] == '03':
                    self.convert_audio_to_spectrogram(file, spectrogram_path)

    def process_savee_files_for_testing(self):
        for subdir, folder, files in os.walk('Savee_Dataset/AudioData/'):
            for file in files:
                savee_emotion = file[0:1] if len(file) == 7 else file[0:2]
                emotion = savee_to_ravdess_emotions[savee_emotion]

                save_path = f'Dataset/test/{emotion}/{file[:-4]}.jpg'
                if f'{file[:-4]}.jpg' not in os.listdir(f'Dataset/test/{emotion}'):
                    self.convert_audio_to_spectrogram(os.path.join(subdir, file), save_path)

f = FileProcessor()
f.process_savee_files_for_testing()
