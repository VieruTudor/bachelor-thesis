import os.path

import tensorflow.keras as keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Rescaling, Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten
from FileProcessor import *
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
dataset_path = f'{os.path.dirname(__file__)}\\Dataset\\Spectrograms\\train'

train_dataset = keras.utils.image_dataset_from_directory(dataset_path,
                                                         label_mode='categorical',
                                                         shuffle=True,
                                                         batch_size=32,
                                                         image_size=(256, 256))

with tf.device('/device:GPU:0'):

    img_path = 'D:\\University\\3rd Year\\Bachelor Thesis\\Project\\test_image.jpg'

    np_image = Image.open(img_path)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = np.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)

    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)

    input_shape = (32, 256, 256, 3)

    print(train_dataset)
    model = keras.Sequential()
    model.add(Rescaling(1. / 255))
    model.add(Dense(256, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(keras.layers.Activation('elu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(keras.layers.Activation('elu'))
    model.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(keras.layers.Activation('elu'))
    model.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(keras.layers.Activation('elu'))
    model.add(MaxPool2D(pool_size=(4, 4), strides=(4, 4)))

    model.add(Flatten())
    model.add(Dense(2048, activation=keras.layers.LeakyReLU(alpha=0.05), input_shape=input_shape))

    model.add(Dense(units=8, activation='softmax'))

    opt = tf.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.build(input_shape=input_shape)

    # model.predict(np_image, batch_size=1)

    print(model.summary())

    # model.fit(train_dataset, epochs=5, verbose=1, callbacks=[checkpoint])

    # TODO: initialize model, use load_weights from best_model.hdf5
    # TODO: to evaluate the model, the steps would be Input (.mp3)
    #  -> Spectrogram
    #  -> Rescale(?)(256, 256)
    #  -> model.predict
