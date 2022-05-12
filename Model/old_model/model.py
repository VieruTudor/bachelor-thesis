import os.path

import tensorflow.keras as keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Rescaling, Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Add, Activation, \
    Layer

from FileProcessor import *
from PIL import Image
import numpy as np
from constants import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.run_functions_eagerly(True)

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

'''
Various notes
TODO: Implementing the model, it will recognize a single emotion from a single unit of work (ie. one sentence with a single emotion).
      Therefore, when evaluating more complex audio files (ie. more sentences -> more emotions), the model should slide between fixed-size
      blocks of frames and evaluate them individually, essentially returning an array of emotions (ex. at second 13, there was happiness
      detected, but at the second 14, sadness was detected, etc.) 
TODO: When converting images to array, try removing trailing black/no sound areas(equivalent to a full-0 matrix)
'''


@tf.keras.utils.register_keras_serializable()
class ResNetBlock(Layer):

    def __init__(self, channels_in=128, kernel=(3, 3), **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.channels_in = channels_in
        self.kernel = kernel

    def call(self, x):
        first_layer = Activation("linear", trainable=False)(x)
        x = Conv2D(self.channels_in,
                   self.kernel,
                   padding="same")(first_layer)
        x = Activation("relu")(x)
        x = Conv2D(self.channels_in,
                   self.kernel,
                   padding="same")(x)
        residual = Add()([x, first_layer])
        x = Activation("relu")(residual)
        return x

    def get_config(self):
        config = super(ResNetBlock, self).get_config()
        config.update({"channels_in": self.channels_in, "kernel": self.kernel})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


dataset_path = f'dataset/BalancedDataset/Dataset/train'

# DataLoader

train_dataset = keras.utils.image_dataset_from_directory(dataset_path,
                                                         label_mode='categorical',
                                                         shuffle=True,
                                                         subset='training',
                                                         validation_split=0.2,
                                                         seed=1161,
                                                         batch_size=32,
                                                         image_size=(256, 256))

validation_dataset = keras.utils.image_dataset_from_directory(dataset_path,
                                                         label_mode='categorical',
                                                         shuffle=True,
                                                         subset='validation',
                                                         validation_split=0.2,
                                                         seed=1337,
                                                         batch_size=32,
                                                         image_size=(256, 256))

def train():
    with tf.device('/device:GPU:0'):
        checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
                                     save_best_only=True, mode='auto', period=1)

        input_shape = (32, 256, 256, 3)

        print(train_dataset)
        model = keras.Sequential()

        pretrained = keras.applications.DenseNet121(input_shape=(256, 256, 3),
                                                    include_top=False,
                                                    weights='imagenet',
                                                    classes=7)
        # pretrained.trainable = False
        model.add(pretrained)
        model.add(Flatten())

        model.add(Dense(128, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Dense(256, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Dense(512, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Dense(1024, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Dense(512, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Dense(256, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Dense(128, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))
        model.add(BatchNormalization())

        model.add(Flatten())
        # model.add(Dense(1024, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))

        model.add(Dropout(rate=0.5))
        model.add(Dense(units=7, activation='softmax'))

        opt = tf.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

        loss = tf.losses.CategoricalCrossentropy(from_logits=False),

        model.compile(optimizer=opt, loss=loss, metrics=['categorical_accuracy'])

        model.build(input_shape=input_shape)

        # print(model.summary())
        model.fit(train_dataset,
                  validation_data=validation_dataset,
                  epochs=100, verbose=1, callbacks=[checkpoint])

def test():
    model_weights = keras
    model = keras.models.load_model('best_model.hdf5', custom_objects={'ResNetBlock': ResNetBlock(128, (3, 3))})
    img_path = '/content/dataset/BalancedDataset/Dataset/train/angry/03-01-05-01-01-01-01.jpg'

    np_image = Image.open(img_path)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = np.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)
    # angry : 2%, happy : 16%, sad : 72%
    res = model.predict(np_image, batch_size=1)[0]

    print([x * 10**3 for x in res])

train()
