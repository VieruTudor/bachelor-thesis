import os

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import pandas

root = os.path.dirname(__file__)
dataset_path = f'{root}\\Dataset'

'''
    File structure : 01 - 01 - 03 - 02 - 02 - 01 - 15
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
'''


class FileProcessor:
    def convert_audio_to_spectrogram(self, file, file_name):
        spectrogram_path = f'{root}\\Dataset\\Spectrograms\\{file_name}.jpg'

        samples, sample_rate = librosa.load(file)
        yt, _ = librosa.effects.trim(samples)

        mel_spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=1024, hop_length=100)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(mel_spectrogram, y_axis='mel', fmax=2 ** 14, x_axis='time')

        plt.savefig(spectrogram_path)
        plt.title('Mel Spectrogram')
        plt.show()

    def process_file(self, file: any):
        file_name, extension = os.path.splitext(file)
        file_name = os.path.basename(file_name)

        # convert .wav to spectrogram
        # we only convert files that are starting with 03 - audio only
        if extension == '.wav' and file_name.split('-')[0] == '03' and file_name + '.jpg' not in os.listdir(
                f'{root}\\Dataset\\Spectrograms'):
            self.convert_audio_to_spectrogram(file, file_name)

    def process_files(self):
        for subdir, dir, files in os.walk(f'{root}\\Dataset\\Audio'):
            for file in files:
                file_path = os.path.join(subdir, file)
                self.process_file(file_path)


    def move_files_to_label_folder(self):

        for subdir, dir, files in os.walk(f'{root}\\Dataset\\Spectrograms'):
            for file in files:
                emotion = file.split('-')[2]
                old_path = f'{root}\\Dataset\\Spectrograms\\{file}'
                new_path = f'{root}\\Dataset\\Spectrograms\\train\\class_{emotion}\\{file}'
                print(old_path)
                print(new_path)
                os.replace(old_path, new_path)


fp = FileProcessor()
fp.move_files_to_label_folder()
