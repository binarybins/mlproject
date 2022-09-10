import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

root_directory = 'Semantic segmentation dataset/'

patch_size = 256

test_directory = '../raw_data/single_image'

image_dataset_test = []
images = sorted(os.listdir(test_directory))
for i, image_name in enumerate(images):
    if image_name.endswith(".jpg"):
        image = cv2.imread(test_directory+"/"+image_name, 1)  #Read each image as BGR
        SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
        SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
        image = Image.fromarray(image)
        image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
        #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
        image = np.array(image)

        #Extract patches from each image
        print("Now patchifying image:", test_directory+"/"+image_name)
        patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):

                single_patch_img = patches_img[i,j,:,:]

                #Use minmaxscaler instead of just dividing by 255.
                single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)

                #single_patch_img = (single_patch_img.astype('float32')) / 255.
                single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.
                image_dataset_test.append(single_patch_img)

image_dataset_test = np.array(image_dataset_test)
image_dataset_test.shape

from keras.models import load_model
model = load_model("../models/satellite_standard_unet_100epochs_8Sep2022.hdf5", compile = False)
y_pred_test=model.predict(image_dataset_test)

for i in range(len(image_dataset_test)):
    test_img = image_dataset_test[i]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img)
    plt.subplot(232)

    plt.title('Prediction on test image')
    plt.imshow(predicted_img)
    plt.show()
