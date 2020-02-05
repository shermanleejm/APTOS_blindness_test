import cv2
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import keras
from keras.utils import to_categorical, np_utils
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint

train_csv = pd.read_csv("train.csv")

train_csv = pd.DataFrame(train_csv)

train_img = []
train_label = []

root = "train/"

for img in os.listdir(root):
    file_name = img.split(".")[0]
    diagnosis =  train_csv.loc[train_csv["id_code"] == file_name, "diagnosis"].iloc[0]
    path = root + img
    
    img = cv2.imread(path)
    train_img.append(img)
    train_label.append(diagnosis)
    # train_data.append([np.array(img), diagnosis])

images = np.array(train_img)
labels = np.array(train_label)

# labels = keras.utils.to_categorical(labels)
labels = np_utils.to_categorical(labels)

path = "train_images"

model = Sequential()

# VGG16 Model
# model.add(Conv2D(input_shape=(300,300,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Flatten())
# model.add(Dense(units=4096,activation="relu"))
# model.add(Dense(units=4096,activation="relu"))
# model.add(Dense(units=5, activation="softmax"))

#random model
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=2, padding='same',activation='relu',input_shape=(300,300,3)))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=32, kernel_size=2, padding='same',activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=64, kernel_size=2, padding='same',activation='relu'))
# model.add(MaxPooling2D())
# model.add(GlobalAveragePooling2D())
# model.add(Dense(5, activation='softmax'))

#random model 2
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=2, padding='same',activation='relu',input_shape=(300,300,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=128, kernel_size=2, padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=256, kernel_size=2, padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=512, kernel_size=2, padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(5, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', 
        optimizer='rmsprop', 
        metrics=['accuracy'])

checkpointer = ModelCheckpoint(
    filepath='saved_models/shermanROX.hdf5', 
    verbose=1, 
    save_best_only=True
    )

model.fit(
    images, 
    labels, 
    batch_size=20, 
    epochs=50, 
    callbacks=[checkpointer], 
    verbose=1,
    validation_split=0.20
    )

model_json = model.to_json()
with open("model.json", "w+") as json_file:
    json_file.write(model_json)

model.save("shermanrox.h5")

# loss, acc = model.evaluate(images, labels, verbose = 0)
# print (acc * 100)
