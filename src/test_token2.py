import numpy as np
import pandas as pd
import json
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

frame_size = 210

def return_real_word(y_pred, map):      # 폴더 순서대로 분류돼있던 클래스를  디렉토리 이름으로 변환해주는 함수
    temp = []
    for v in y_pred:
        temp.append(map[v])
    return np.array(temp)

X_train = np.load("../resource/X_train.npy")
X_test = np.load("../resource/X_test.npy")
y_train = np.load("../resource/y_train.npy")
y_test = np.load("../resource/y_test.npy")

with open('train_data_map.json', 'r') as json_file:          ### 딕셔너리 불러오기
    train_data_map = json.load(json_file)

with open('test_data_map.json', 'r') as json_file:          ### 딕셔너리 불러오기
    test_data_map = json.load(json_file)

length = len(train_data_map)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(70, 50, 3), activation='swish'))
model.add(Conv2D(128, (3, 3), activation='swish' ))
model.add(Conv2D(64, (3, 3), activation='swish'))
# model.add(Conv2D(32, (3, 3), activation='swish'))
model.add(Conv2D(frame_size, (3, 3), activation='swish'))
model.add(Reshape(target_shape=(frame_size, -1)))
model.add(LSTM(32,))
model.add(Dense(64, activation='swish'))
model.add(Dense(32, activation='swish'))
model.add(Dense(length, activation='softmax'))
# batch_normalizaton 추가 예정
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
# es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
es = EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
model.fit(X_train, y_train, epochs=5, batch_size=10, shuffle=False, callbacks=[es])
# model.fit(X_train, y_train, epochs=30, batch_size=42, shuffle=False, validation_data=(X_test, y_test), validation_batch_size=42, callbacks=[es])


y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

def seperate_pred(y_pred, fs):
    for i in range(int(len(y_pred) / fs)):
        y_pred[i * fs : (i+1) * fs] = select_pred(y_pred[i * fs + int(fs/2) : (i+1) * fs])      # 영상 앞은 패딩되있기 때문에, 프레임의 후반부 50%의 예측값으로 모든 예측값을 대체
    return y_pred

def select_pred(y_pred):
    temp = []
    for i in y_pred:
        temp.append(i)
    x = most_common_element(temp)      # frame 단위로 예측을 하다보니 영상 한개에서 여러가지 라벨이 예측되는데, 그 중 가장 많이 예측된 라벨값을 반환
    return x

def most_common_element(arr):
    # Counter를 사용하여 배열 안의 요소들의 빈도수를 계산
    counter = Counter(arr)
    # most_common 메서드를 사용하여 가장 많이 등장하는 요소와 빈도수를 가져옴
    most_common = counter.most_common(1)
    if most_common:
        return most_common[0][0]  # 가장 많이 등장하는 요소 반환
    else:
        return None  # 빈 배열이면 None 반환

print("++++++++++++++++++++++++++++++++++++++++++++")

y_pred = seperate_pred(y_pred, frame_size)
print("+=========================================")
y_pred = return_real_word(y_pred, train_data_map)
y_test = return_real_word(y_test, test_data_map)
print("+=========================================")



print("ypred:",y_pred)
print("++++++++++++++++++++++++++++++++++++++++++++")

print("ytest:",y_test)
# print(np.unique(y_pred, return_counts=True))



acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time), "초")
