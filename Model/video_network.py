import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras import Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import *
import cv2

from constants import emotions, fer_to_emotions

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
    df = pd.read_csv('FER_Dataset/fer2013.csv')
    training = df.loc[df['Usage'] == 'Training']
    testing = df.loc[df['Usage'] != 'Training']

    # X_train values:
    X_train = training[['pixels']].values
    X_train = [np.fromstring(e[0], dtype=int, sep=' ') / 255.0 for e in X_train]
    if net:
        X_train = [e.reshape((48, 48, 1)).astype('float32') for e in X_train]
    else:
        X_train = [e.reshape((48, 48)) for e in X_train]
    X_train = np.array(X_train)

    # X_test values:
    X_test = testing[['pixels']].values
    X_test = [np.fromstring(e[0], dtype=int, sep=' ') / 255.0 for e in X_test]
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


def train():
    with tf.device('/device:CPU:0'):
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

        print(class_weight)

        y_train = np.array(y_train)

        model = Sequential()

        """
        Convolution and Maxpool layers: Block 1
        """
        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='linear', input_shape=(48, 48, 1)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))

        model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))

        # MaxPool layer: 1
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.3))

        """
        Convolution and Maxpool layers: Block 2
        """
        # Conv Layer 3:24x24x64
        model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))

        # Conv Layer 4:24x24x64
        model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))

        # MaxPool layer: 2
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.3))

        """
        Convolution and Maxpool layers: Block 3
        """
        # Conv Layer 5:12x12x128
        model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))

        # Conv Layer 6:12x12x128
        model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))

        # MaxPool layer: 3
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))

        """
        Convolution and Maxpool layers: Block 4
        """
        # Conv Layer 7:6x6x256
        model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))

        # Conv Layer 8:6x6x256
        model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))

        # MaxPool layer: 4
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))

        # Flatten
        model.add(Flatten())

        # Dense layer 1:
        model.add(Dense(256, activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.5))

        # Dense layer 2:
        model.add(Dense(256, activation='linear'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(0.5))

        model.add(Dense(units=7, activation='softmax'))

        print(model.summary())
        return


        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
        history = model.fit(
            # X_train.reshape(-1, 48, 48, 1),
            X_train,
            y_train,
            batch_size=32,
            epochs=25,
            validation_data=(X_test, y_test),
            class_weight=class_weight)

        plot_accuracy(history)
        plot_losses(history)

        model.save("video_model.h5")


def test():
    with tf.device('/device:CPU:0'):
        file_name = 'Video_Tests/angry.jpg'
        file_name_only = file_name[file_name.rfind('/') + 1:]
        print(file_name)
        print(file_name_only)
        img = np.array(Image.open(file_name), dtype=np.uint8)

        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)

            faces = img[y:y + h, x:x + w]

        cv2.waitKey()

        img = Image.fromarray(faces)
        img = img.resize((48, 48))
        img = np.array(img, dtype='float32')
        img /= 255.0
        img = img.reshape((1, 48, 48))

        model = tf.keras.models.load_model('vgg_2nd_try.h5')
        results = model.predict(img)[0]

        print([x * 100 for x in results])
        final_preds = []
        results = [x * 100 for x in results]
        for result in range(len(results)):
            final_preds.append(emotions[fer_to_emotions[result]])
        plt.bar(final_preds, results)
        plt.ylabel('Level (%)')
        plt.xlabel('Emotion')
        plt.savefig('angry_bar_plot.png')

        plt.show()
test()
