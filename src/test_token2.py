import numpy as np
import pandas as pd
import cv2
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, TimeDistributed, LSTM, Flatten, Reshape
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, accuracy_score
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from collections import Counter

start_time = time.time()

image_path = "../resource/test_pad/train/"

imgdatagen = ImageDataGenerator(
    rescale=1/255.
    
)
frame_size = 210



test_data = imgdatagen.flow_from_directory(
                "../resource/test_pad/test/",
                target_size=(70, 50),
                batch_size=99999,
                class_mode='sparse',
                shuffle=False,
)   # data[몇번째 batch][0 : X값, 1 : 라벨(y) 값][frame_num]


X_test = test_data[0][0]
y_test = test_data[0][1]

import matplotlib.pyplot as plt
plt.imshow(X_test[0])
plt.show()


end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time), "초")
