from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import decode_predictions as decode_predictions_res
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_res
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_predictions_vgg
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
import keras.utils as kimage

import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os

img_size = 224

# load images
images_folder = "./experiment/"
image_names = [e for e in os.listdir(images_folder) if ".JPG" in e.upper()]
image_names.sort()
num_images = len(image_names)
all_images = np.zeros((num_images, img_size, img_size, 3))
dims = np.zeros((num_images, 2))
for i in range(num_images):
    image = kimage.load_img(images_folder + image_names[i])
    image = kimage.img_to_array(image)
    dims[i] = [float(image.shape[0]), float(image.shape[1])]
    image = cv2.resize(image, (img_size, img_size))
    all_images[i] = image


image = tf.constant(all_images)

# run through models
resnet = ResNet50(weights="imagenet")
vgg = VGG16(weights="imagenet")

print("Running ResNet50")
preprocessedImage_res = preprocess_input_res(image)
predictions_res = resnet.predict(preprocessedImage_res)
predictions_res = decode_predictions_res(predictions_res, top=3)

print("Running VGG16:")
preprocessedImage_vgg = preprocess_input_vgg(image)
predictions_vgg = vgg.predict(preprocessedImage_vgg)
predictions_vgg = decode_predictions_res(predictions_vgg, top=3)

# output to csv file
df = pd.DataFrame(columns=["ResNet Label", "ResNet Confidence",
                           "VGG Label", "VGG Confidence"])

for i in range(num_images):

    row = {"ResNet Label": predictions_res[i][0][1], "ResNet Confidence": predictions_res[i][0][2],
           "VGG Label": predictions_vgg[i][0][1], "VGG Confidence": predictions_vgg[i][0][2]}
    df.loc[image_names[i]] = pd.Series(row)

df.to_csv("results.csv")
