import numpy as np
import pandas as pd
import cv2
import os
import time
from keras.models import Sequential
from keras.layers import Dense, Conv2D, TimeDistributed, LSTM, Flatten
start_time = time.time()
def extract_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    channels = 3  # Assuming RGB

    cap.release()

    return {'video_path': video_path, 'frames': frames, 'height': height, 'width': width, 'channels': channels}

# 디렉토리 내의 모든 영상 파일에 대한 정보 추출
video_directory = '../resource/수어/영상/'
video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.mts') or f.endswith('.mov')]

video_info_list = []
for video_file in video_files:
    video_path = os.path.join(video_directory, video_file)
    video_info = extract_video_info(video_path)
    video_info_list.append(video_info)

# DataFrame 생성
video_df = pd.DataFrame(video_info_list)
print(video_df.shape)
print(video_df)


end_time = time.time()

print("걸린 시간 : ", round(end_time - start_time), "초")