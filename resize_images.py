import cv2
import pandas as pd 
import numpy as np
import os

count = 0

where = "train"

train_set = pd.read_csv(f"{where}.csv")

path = where + "_images"

for image in os.listdir(path):
    count += 1
    print (count)

    file_name = image.split(".")[0]
    # severity =  train_set.loc[train_set["id_code"] == file_name, "diagnosis"].iloc[0]
    file_path = path + "/" + image
    dim = (300, 300)

    img = cv2.imread(file_path)

    img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    out_path = f"{where}/" + image

    cv2.imwrite(out_path, img_resize) 

