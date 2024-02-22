import numpy as np
import pandas as pd
import cv2
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, TimeDistributed, LSTM, Flatten
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

start_time = time.time()

image_path = "../resource/test_pad/"

imgdatagen = ImageDataGenerator(
    rescale=1/255.
)

data = imgdatagen.flow_from_directory(
                image_path,
                target_size=(1280, 720),
                batch_size=150,
                class_mode='categorical',
                shuffle=False,
)

print(data[0][0].shape)
plt.imshow(data[2][0][145])     # data[몇번째 class][몇번째 배치][frame]
# plt.imshow(data[0][0][1])
plt.show()


end_time = time.time()

print("걸린 시간 : ", round(end_time - start_time), "초")