import numpy as np
import pandas as pd
import json
import cv2
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, TimeDistributed, LSTM, Flatten, Reshape, GRU
from keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from keras import backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # GPU 메모리 증가 허용
session = tf.compat.v1.Session(config=config)

start_time = time.time()
image_path = "../resource/test_pad/train/"

frame_size = 210
sequence_len = frame_size
def return_real_word(y_pred, map):      # 폴더 순서대로 분류돼있던 클래스를  디렉토리 이름으로 변환해주는 함수
    temp = []
    for v in y_pred:
        temp.append(map[v])
    return np.array(temp)

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

def create_model():
    CNN_model = Sequential()
    CNN_model.add(Conv2D(64, (3, 3), input_shape=(70, 50, 3), activation='swish'))
    CNN_model.add(Conv2D(128, (3, 3), activation='swish' ))
    CNN_model.add(Conv2D(64, (3, 3), activation='swish'))
    # CNN_model.add(Conv2D(frame_size, (3, 3), activation='swish'))

    time_dist_CNN_model = Sequential()
    time_dist_CNN_model.add(TimeDistributed(CNN_model, input_shape=(sequence_len, 70, 50, 3)))

    model = Sequential()
    model.add(time_dist_CNN_model)              #  이대로 하면 LSTM에 입력되는 데이터가 (batch_size, timesteps, height, width, chanels) (5차원)
    model.add(TimeDistributed(Flatten()))       #  LSTM에 입력되는 데이터의 형태가 (batch_size, timesteps, features) (3차원)
    model.add(GRU(64,))
    model.add(Dense(32, activation='swish'))
    model.add(Dense(128, activation='swish'))
    model.add(Dense(64, activation='swish'))
    model.add(Dense(length, activation='softmax'))
    # batch_normalizaton 추가 예정
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def ensemble(models, data):
    predictions = [model.predict(data) for model in models]
    return np.mean(predictions, axis=0) # softmax의 확률들을 평균

X_train = np.load("C:\_data\_save_npy\\X_train.npy")
X_test = np.load("C:\_data\_save_npy\\X_test.npy")
y_train = np.load("C:\_data\_save_npy\\y_train.npy")
y_test = np.load("C:\_data\_save_npy\\y_test.npy")
with open('C:\\Users\\user\\Desktop\\test_pad\\train_data_map.json', 'r') as json_file:          ### 딕셔너리 불러오기
    train_data_map = json.load(json_file)
with open('C:\\Users\\user\\Desktop\\test_pad\\test_data_map.json', 'r') as json_file:          ### 딕셔너리 불러오기
    test_data_map = json.load(json_file)
print("data load end")
length = len(train_data_map)



X_1, X_2, y_1, y_2 = train_test_split(X_train, y_train, stratify=y_train, random_state=42, test_size=0.5)
X11, X12, y11, y12 = train_test_split(X_1, y_1, stratify=y_1, random_state=42, test_size=0.5)
X13, X14, y13, y14 = train_test_split(X_2, y_2, stratify=y_2, random_state=42, test_size=0.5)

X1, X2, y1, y2 = train_test_split(X11, y11, stratify=y11, random_state=42, test_size=0.5)
X3, X4, y3, y4 = train_test_split(X12, y12, stratify=y12, random_state=42, test_size=0.5)
X5, X6, y5, y6 = train_test_split(X13, y13, stratify=y13, random_state=42, test_size=0.5)
X7, X8, y7, y8 = train_test_split(X14, y14, stratify=y14, random_state=42, test_size=0.5)


# X1, X2, X3, X4, y1, y2, y3, y4 = train_test_split(X1, X2, y1, y2, stratify=y_train, random_state=42, test_size=0.5)

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

model1 = create_model()
model1.fit(X1, y1, epochs=5, batch_size=5, shuffle=True, callbacks=[es], validation_split=0.2)
K.clear_session()
model2 = create_model()
model2.fit(X2, y2, epochs=5, batch_size=5, shuffle=True, callbacks=[es], validation_split=0.2)
K.clear_session()
model3 = create_model()
model3.fit(X3, y3, epochs=5, batch_size=5, shuffle=True, callbacks=[es], validation_split=0.2)
K.clear_session()
model4 = create_model()
model4.fit(X4, y4, epochs=5, batch_size=5, shuffle=True, callbacks=[es], validation_split=0.2)
K.clear_session()
model5 = create_model()
model5.fit(X5, y5, epochs=5, batch_size=5, shuffle=True, callbacks=[es], validation_split=0.2)
K.clear_session()
model6 = create_model()
model6.fit(X6, y6, epochs=5, batch_size=5, shuffle=True, callbacks=[es], validation_split=0.2)
K.clear_session()
model7 = create_model()
model7.fit(X7, y7, epochs=5, batch_size=5, shuffle=True, callbacks=[es], validation_split=0.2)
K.clear_session()
model8 = create_model()
model8.fit(X8, y8, epochs=5, batch_size=5, shuffle=True, callbacks=[es], validation_split=0.2)
K.clear_session()

model1.save('model1.h5')
model2.save('model2.h5')
model3.save('model3.h5')
model4.save('model4.h5')
model5.save('model5.h5')
model6.save('model6.h5')
model7.save('model7.h5')
model8.save('model8.h5')


y_pred = ensemble([model1, model2, model3, model4, model5, model6, model7, model8], X_test)
y_pred = np.argmax(y_pred, axis=1)



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