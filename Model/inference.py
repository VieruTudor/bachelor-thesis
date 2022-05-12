import os
import sys
import tensorflow as tf
import librosa
import numpy as np

from constants import SUPPORTED_AUDIO_FILES, SUPPORTED_VIDEO_FILES, SAMPLE_RATE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file_path = sys.argv[1]
extension = os.path.splitext(file_path)[1]


def process_audio():
    with tf.device('/device:GPU:0'):
        model = tf.keras.models.load_model('D:/University/BachelorThesis/Model/dense_84_train_71_val.h5')

        audio_path = file_path

        audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio = librosa.effects.trim(audio)[0]

        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=40).T, axis=0)
        spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=48000, n_mels=40).T, axis=0)
        x = np.asarray((mfcc, spectrogram), dtype=np.float32)
        x = np.reshape(x, (1, 2, 40))

        predictions = model.predict(x)[0]
        preds = ""
        for i in range(len(predictions)):
            preds += str(i) + " " + str(predictions[i] * 100) + ";"

        sys.stdout.write(preds)


def process_video():
    pass


if extension in SUPPORTED_AUDIO_FILES:
    process_audio()
    sys.exit()
elif extension in SUPPORTED_VIDEO_FILES:
    process_video()
    sys.exit()
else:
    process_audio()
    process_video()
    sys.exit()
