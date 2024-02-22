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
import matplotlib.pyplot as plt
from collections import Counter

start_time = time.time()

image_path = "../resource/test_pad/train/"

imgdatagen = ImageDataGenerator(
    rescale=1/255.
    
)
frame_size = 150

train_data = imgdatagen.flow_from_directory(
                image_path,
                target_size=(70, 50),
                batch_size=99999,
                class_mode='sparse',
                shuffle=False,
)   # data[몇번째 batch][0 : X값, 1 : 라벨(y) 값][frame_num]

test_data = imgdatagen.flow_from_directory(
                "../resource/test_pad/test/",
                target_size=(70, 50),
                batch_size=99999,
                class_mode='sparse',
                shuffle=False,
)   # data[몇번째 batch][0 : X값, 1 : 라벨(y) 값][frame_num]

# plt.imshow(data[0][0][148])     # data[몇번째 batch][0 : X값, 1 : 라벨(y) 값][frame_num]
# plt.show()
# for i in range(int(745 / frame_size)):
#     X[i], y[i] = data[i][0], data[i][1]

X_train = train_data[0][0]
y_train = train_data[0][1]
X_test = test_data[0][0]
y_test = test_data[0][1]



# plt.imshow(X[148])     # data[몇번째 batch][0 : X값, 1 : 라벨(y) 값][frame_num]
# plt.show()

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
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=frame_size, ) #shuffle=False)



res = model.evaluate(X_test, y_test)
print("acc : ", res[1])

y_pred = model.predict(X_test)
print(y_pred)
y_pred = np.argmax(y_pred, axis=1)
print(np.unique(y_pred, return_counts=True))

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

print(y_pred)
print(np.unique(y_pred, return_counts=True))

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time), "초")