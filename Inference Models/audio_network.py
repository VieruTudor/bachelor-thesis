import os

import joblib
import librosa
import numpy as np
import tensorflow as tf

from keras import Sequential, regularizers
from keras.applications.resnet import ResNet50
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import *
from sklearn.svm import SVC

from constants import *

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

checkpoint = ModelCheckpoint("best_model_checkpoint.hdf5", monitor='loss', verbose=1,
                             save_best_only=True, mode='auto', period=1)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.run_functions_eagerly(True)


class ValidationShuffle(Callback):

    def __init__(self, x_test):
        super().__init__()
        self.x_test = x_test

    def on_epoch_end(self, epoch, logs=None):
        np.random.shuffle(self.x_test)


def plot_losses(train_history):
    plt.plot(train_history.history['loss'])
    # plt.plot(train_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png')
    plt.close()


def plot_accuracy(train_history):
    plt.plot(train_history.history['sparse_categorical_accuracy'])
    # plt.plot(train_history.history['val_sparse_categorical_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')


def train():
    with tf.device('/device:GPU:0'):
        x_train = joblib.load("x_train_split.joblib")
        y_train = joblib.load("y_train_split.joblib")

        x_test = joblib.load("x_test_split.joblib")
        y_test = joblib.load("y_test_split.joblib")

        # x_train = np.reshape(x_train, (2, 40,))
        # x_test = np.reshape(x_test, (40, 2,))

        model = Sequential()

        model.add(Input(shape=(2, 40,)))

        model.add(Dense(40, activation='relu'))

        model.add(Dense(80, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(64, activation='relu'))

        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))

        model.add(Dense(32, activation='relu'))

        model.add(Flatten())

        model.add(Dropout(rate=0.1))
        model.add(Dense(units=7, activation='softmax'))

        opt = tf.optimizers.Adam(lr=0.0001)

        loss = tf.losses.SparseCategoricalCrossentropy(),

        model.compile(optimizer=opt, loss=loss, metrics=['sparse_categorical_accuracy'])

        model.build(input_shape=(40, 2,))

        history = model.fit(x=x_train,
                            y=y_train,
                            validation_data=(x_test, y_test),
                            epochs=20,
                            batch_size=16,
                            validation_batch_size=16,
                            verbose=1)

        plot_accuracy(history)
        plot_losses(history)

        model.save("new_model.h5")


def test():
    with tf.device('/device:CPU:0'):

        model = tf.keras.models.load_model('dense_84_train_71_val.h5')
        print(model.summary())
        return

        audio_path = 'C:/Users/vieru/Downloads/Happy (1).wav'

        audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio = librosa.effects.trim(audio)[0]

        mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=40).T, axis=0)
        spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=48000, n_mels=40).T, axis=0)
        x = np.asarray((mfcc, spectrogram), dtype=np.float32)
        x = np.reshape(x, (1, 2, 40))

        predictions = model.predict(x)[0]
        print(predictions)
        for i in range(len(predictions)):
            print(emotions[i], predictions[i] * 100)

        # x_test = joblib.load("x_test_split.joblib")
        # y_test = joblib.load("y_test_split.joblib")
        #
        # x_test = np.reshape(x_test, (len(x_test), 2, 40))
        # results = model.evaluate(x=x_test,
        #                          y=y_test,
        #                          batch_size=16,
        #                          verbose=1)
        # print(results)
        #
        # y_pred = model.predict(x_test)
        # print(y_pred)
        #
        # # y_pred = np.argmax(y_pred, axis=1)
        # # print(y_pred)
        # # return
        #
        # y_pred = [tf.math.top_k(y, k=2).indices.numpy() for y in y_pred]
        # print(y_pred)
        #
        # for i in range(len(y_pred)):
        #     if y_test[i] in y_pred[i]:
        #         y_pred[i] = y_test[i]
        #     else:
        #         y_pred[i] = np.argmax(y_pred[i])
        # print(y_pred)
        #
        # y_test = [emotions[a] for a in y_test]
        # y_pred = [emotions[a] for a in y_pred]
        #
        # conf_mat = confusion_matrix(y_test, y_pred)
        # ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=list(emotions.values()), )
        # print(np.average(conf_mat.diagonal() / conf_mat.sum(axis=1)))
        #
        # plt.savefig('conf_matrix_top3.png')


test()
