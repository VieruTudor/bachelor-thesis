import os

import joblib
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd

from keras import Sequential, regularizers, Model
from keras.applications.resnet import ResNet50
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import *
from keras.regularizers import l2
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

from constants import FER_PATH
from PIL.Image import Image

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


def load_dataset(net=True):
    # Load and filter in Training/not Training data:
    df = pd.read_csv('fer2013.csv')
    training = df.loc[df['Usage'] == 'Training']
    testing = df.loc[df['Usage'] != 'Training']

    # X_train values:
    X_train = training[['pixels']].values
    X_train = [np.fromstring(e[0], dtype=int, sep=' ') / 255 for e in X_train]
    if net:
        X_train = [e.reshape((48, 48, 1)).astype('float32') for e in X_train]
    else:
        X_train = [e.reshape((48, 48)) for e in X_train]
    X_train = np.array(X_train)

    # X_test values:
    X_test = testing[['pixels']].values
    X_test = [np.fromstring(e[0], dtype=int, sep=' ') / 255 for e in X_test]
    if net:
        X_test = [e.reshape((48, 48, 1)).astype('float32') for e in X_test]
    else:
        X_test = [e.reshape((48, 48)) for e in X_test]
    X_test = np.array(X_test)

    # y_train values:
    y_train = [val[0] for val in training[['emotion']].values]
    # y_train = tf.keras.utils.to_categorical(y_train)

    # y_test values
    y_test = [val[0] for val in testing[['emotion']].values]
    # y_test = tf.keras.utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def entry_flow(inputs):
    x = Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation = x

    for size in [64, 128, 256]:
        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding='same')(x)

        residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)

        x = tf.keras.layers.Add()([x, residual])
        previous_block_activation = x

    return x


def middle_flow(x, num_blocks=8):
    previous_block_activation = x

    for _ in range(num_blocks):
        x = Activation('relu')(x)
        x = SeparableConv2D(256, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(256, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(256, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = tf.keras.layers.Add()([x, previous_block_activation])
        previous_block_activation = x

    return x


def exit_flow(x):
    previous_block_activation = x

    x = Activation('relu')(x)
    x = SeparableConv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=2, padding='same')(x)

    residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = tf.keras.layers.Add()([x, residual])

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(7, activation='softmax', activity_regularizer=l2(0.001))(x)

    return x


def train():
    with tf.device('/device:GPU:0'):
        X_train, y_train, X_test, y_test = load_dataset()
        print(X_train.shape)

        class_weight = {
            0: 1 / y_train.count(0),
            1: 1 / y_train.count(1),
            2: 1 / y_train.count(2),
            3: 1 / y_train.count(3),
            4: 1 / y_train.count(4),
            5: 1 / y_train.count(5),
            6: 1 / y_train.count(6)
        }

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        inputs = Input(shape=(48, 48, 1))

        outputs = exit_flow(middle_flow(entry_flow(inputs)))
        model = Model(inputs, outputs)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        history = model.fit(
            # X_train.reshape(-1, 48, 48, 1),
            X_train,
            y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            class_weight=class_weight)

        plot_accuracy(history)
        plot_losses(history)

        model.save("video_model.h5")


def test():
    with tf.device('/device:CPU:0'):
        image = Image.Load("test.jpg")
        image.re


train()
