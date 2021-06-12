import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
import random, shutil
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1. / 255), shuffle=True, batch_size=1, target_size=(24, 24),
              class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale',
                                   class_mode=class_mode, target_size=target_size)


batch_size = 32
total_size = (24, 24)
train_batch = generator('data/test', shuffle=True, batch_size=batch_size, target_size=total_size)
valid_batch = generator('data/valid', shuffle=True, batch_size=batch_size, target_size=total_size)
batch_classes = len(train_batch.classes) // batch_size
valid_classes = len(valid_batch.classes) // batch_size
print(batch_classes, valid_classes)


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=batch_classes , validation_steps=valid_classes)

model.save('models/cnnCat2.h5', overwrite=True)