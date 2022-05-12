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
        X = joblib.load(X_TRAIN_DATA)
        y = joblib.load(Y_TRAIN_DATA)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples, nx * ny))

        nsamples, nx, ny = X_test.shape
        X_test = X_test.reshape((nsamples, nx * ny))

        # Simple SVM
        print('fitting...')
        clf = SVC(C=20.0, gamma=0.00001)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print("acc=%0.3f" % acc)

        # Grid search for best parameters
        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
                             'C': [1, 10 ,20,30,40,50]}]
                            #  ,
                            # {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print('')

            clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                               scoring='%s_macro' % score)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print('')
            print(clf.best_params_)
            print('')
            print("Grid scores on development set:")
            print('')
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print('')

            print("Detailed classification report:")
            print('')
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print('')
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print('')


def test():
    with tf.device('/device:GPU:0'):
        # model = tf.keras.models.load_model('C:/Users/vieru/Desktop/Models/model_28_acc.h5')
        model = tf.keras.models.load_model('pretrained_model.h5')
        # audio_path = 'C:/Users/vieru/Downloads/angry.wav'
        #
        # audio, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
        # audio = librosa.effects.trim(audio)[0]
        #
        # mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=48000, n_mfcc=40).T, axis=0)
        # spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=48000, n_mels=40).T, axis=0)
        # x = np.asarray((mfcc, spectrogram), dtype=np.float32)
        # x = np.reshape(x, (1, 40, 2))
        #
        # predictions = model.predict(x)[0]
        # print(predictions)
        # for i in range(len(predictions)):
        #     print(emotions[i], predictions[i] * 100)

        x_test = joblib.load(X_TEST_DATA)
        y_test = joblib.load(Y_TEST_DATA)

        print(len(x_test))

        x_test = np.reshape(x_test, (len(x_test), 40, 2))
        results = model.evaluate(x=x_test,
                                 y=y_test,
                                 batch_size=8,
                                 verbose=1)
        print(results)

        y_pred = model.predict(x_test)
        print(y_pred)

        # y_pred = np.argmax(y_pred, axis=1)
        # print(y_pred)
        # return

        y_pred = [tf.math.top_k(y, k=3).indices.numpy() for y in y_pred]
        print(y_pred)

        for i in range(len(y_pred)):
            if y_test[i] in y_pred[i]:
                y_pred[i] = y_test[i]
            else:
                y_pred[i] = np.argmax(y_pred[i])
        print(y_pred)

        y_test = [emotions[a] for a in y_test]
        y_pred = [emotions[a] for a in y_pred]

        conf_mat = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=list(emotions.values()), )
        print(np.average(conf_mat.diagonal() / conf_mat.sum(axis=1)))

        plt.show()
        plt.savefig('conf_matrix_top3.jpg')


train()
