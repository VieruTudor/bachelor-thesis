import os
import sys
import tensorflow as tf
import librosa
import numpy as np
import ast

from constants import SUPPORTED_AUDIO_FILES, SUPPORTED_VIDEO_FILES, SAMPLE_RATE
from flask import Flask, request
from flask_restful import Resource, Api


def initialise_model():
    global model
    with tf.device('/device:GPU:0'):
        model = tf.keras.models.load_model('D:/University/BachelorThesis/Model/dense_84_train_71_val.h5')


initialise_model()

app = Flask(__name__)
api = Api(app)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# file_path = sys.argv[1]
# extension = os.path.splitext(file_path)[1]

global model


def process_audio(file_path):
    global model
    audio_path = file_path

    audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio = librosa.effects.trim(audio)[0]

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=40).T, axis=0)
    spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=48000, n_mels=40).T, axis=0)
    x = np.asarray((mfcc, spectrogram), dtype=np.float32)
    x = np.reshape(x, (1, 2, 40))

    predictions = model.predict(x)[0]
    preds = {}
    for i in range(len(predictions)):
        preds[i] = predictions[i] * 100

    return preds

def process_video():
    pass


@app.route('/detect-emotion-from-audio', methods=['POST'])
def detect_emotion_from_audio():
    print(type(request.get_json()))

    print("request", request.get_json())
    file_path = ast.literal_eval(request.get_json())['AudioPath']
    results = process_audio(file_path)
    print(results)
    return {'results': results}


if __name__ == '__main__':
    app.run(port=3000, debug=True)
