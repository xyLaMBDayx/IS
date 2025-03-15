import os
import xml.etree.ElementTree as ET
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import cv2
import numpy as np

# Convert to grayscale
x_train_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train])
x_test_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_test])

import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

model = models.Sequential([
    layers.Input(shape=(32, 32, 1)),

    layers.Conv2D(128, (3, 3), padding='same'),
    layers.LeakyReLU(negative_slope =0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3, 3), padding='same'),
    layers.LeakyReLU(negative_slope =0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(512, (3, 3), padding='same'),
    layers.LeakyReLU(negative_slope =0.1),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.GlobalMaxPooling2D(),

    layers.Dense(512),
    layers.LeakyReLU(negative_slope =0.1),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128),
    layers.LeakyReLU(negative_slope =0.1),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(2, activation='softmax')
])

# **Adaptive Learning Rate**
optimizer = Adam(learning_rate=0.000001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# **Callbacks for Better Training**
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),  # Reduce LR if loss plateaus
]

# **Train the Model**
model.fit(x_train_gray, y_train, epochs=50, validation_data=(x_test_gray, y_test), callbacks=callbacks)
