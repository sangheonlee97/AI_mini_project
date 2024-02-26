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

def split_X(data, frame_size):
    X = []
    for i in range(int(len(data) / frame_size) ):
        X_s = data.iloc[i * frame_size : (i+1) * frame_size]
        X.append(X_s)
    return X

def split_y(data, frame_size):
    y = []
    for i in range(int(len(data) / frame_size) ):
        y_s = data.iloc[i * frame_size]
        y.append(y_s)
    return y
def split(X_data, y_data, frame_size):
    return split_X(X_data, frame_size), split_y(y_data, frame_size)

train_data = imgdatagen.flow_from_directory(
                "../resource/test_pad/train/",
                target_size=(70, 50),
                batch_size=frame_size,
                class_mode='sparse',
                shuffle=False, # 나중에 배치 단위로 셔플해야 하는데..
)   # data[몇번째 batch][0 : X값, 1 : 라벨(y) 값][frame_num]


test_data = imgdatagen.flow_from_directory(
                "../resource/test_pad/test/",
                target_size=(70, 50),
                batch_size=frame_size,
                class_mode='sparse',
                shuffle=False,
)   # data[몇번째 batch][0 : X값, 1 : 라벨(y) 값][frame_num]


X_train = []
y_train = []
for i in range(len(train_data)):
    batch = train_data.next()
    X_train.append(batch[0])
    y_train.append(batch[1])
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
X_train = X_train.reshape(-1 , 70 * 50 * 3)
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_train, y_train = split(X_train, y_train, frame_size)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train.reshape(-1, frame_size ,70, 50, 3)

print("train data")
print(X_train.shape, y_train.shape)


X_test = []
y_test = []
for i in range(len(test_data)):
    batch = test_data.next()
    X_test.append(batch[0])
    y_test.append(batch[1])
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)
X_test = X_test.reshape(-1 , 70 * 50 * 3)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)
X_test, y_test = split(X_test, y_test, frame_size)
X_test = np.array(X_test)
y_test = np.array(y_test)
X_test = X_test.reshape(-1, frame_size ,70, 50, 3)

print("test data")
print(X_test.shape, y_test.shape)


np.save( "../resource/X_train.npy", X_train)
np.save( "../resource/X_test.npy", X_test)
np.save( "../resource/y_train.npy", y_train)
np.save( "../resource/y_test.npy", y_test)

end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time), "초")
