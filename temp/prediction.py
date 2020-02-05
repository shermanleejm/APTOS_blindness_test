from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import os
from keras.models import Sequential, model_from_json

# json_file = open("./model.json", 'r')
# json_model = json_file.read()
# json_file.close()
# model = model_from_json(json_model)

# model.load_weights('shermanROX.hdf5')

model = load_model('shermanROX.h5')

model.compile(  
    loss='categorical_crossentropy', 
    optimizer='rmsprop', 
    metrics=['accuracy']    
)

root = "test/"

images = []

for img in os.listdir(root):
    name = img.split(".")
    path = root + img

    img = cv2.imread(path)
    images.append(img)

images = np.array(images)

classes = model.predict_classes(images)

df = pd.DataFrame(classes)

output = df.to_csv("classes.csv", mode="a")