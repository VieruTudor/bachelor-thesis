import os.path

import PIL.Image as Image
import numpy as np
from tensorflow.keras import layers
import torch
import tensorflow as tf
import tensorflow.keras.applications as tfapp
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
from torch.utils.data.dataloader import DataLoader
from torchvision import *
from fastai.vision.all import *
from pathlib import Path
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
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# img_arr = []
# for subdir, dirs, files in os.walk(dataset_path):
#     for file in files:
#         file_path = os.path.join(subdir, file)
#         img = np.array(Image.open(file_path))
#         img = np.resize(img, (256, 256, 3))
#         img = img.astype('float32')
#         img /= 255
#         print(img)
#         img_arr.append(img)

train_dataset = keras.utils.image_dataset_from_directory(dataset_path,
                                                         shuffle=True,
                                                         batch_size=64,
                                                         image_size=(256, 256))

for images, labels in train_dataset.take(1):  # only take first element of dataset
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    print(numpy_images, numpy_labels)

base_model = keras.applications.EfficientNetB4(input_shape=(256, 256, 3),
                                         include_top=False,
                                         weights='imagenet')



image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

inputs = keras.Inputs(shape=(255,255,3))
x = keras.Sequential()
x = layers.Rescaling(1./255)(x)
