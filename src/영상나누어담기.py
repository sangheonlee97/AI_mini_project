import os
import shutil
from sklearn.model_selection import train_test_split

def split_data_by_label(data_dir, train_dir, test_dir, test_size=0.2, random_state=42):
    # data_dir: 라벨별로 분류된 동영상이 있는 최상위 디렉토리
    # train_dir: Train 데이터를 저장할 디렉토리
    # test_dir: Test 데이터를 저장할 디렉토리

    # 모든 라벨 디렉토리 가져오기
    labels = os.listdir(data_dir)

    # Train, Test 디렉토리 생성
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 라벨별로 데이터 수집 및 나누기
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            videos = [os.path.join(label_dir, video) for video in os.listdir(label_dir) if video.endswith('.mp4')]
            X_train, X_test = train_test_split(videos, test_size=test_size, random_state=random_state)
            # Train 데이터 이동
            train_label_dir = os.path.join(train_dir, label)
            os.makedirs(train_label_dir, exist_ok=True)
            for train_video in X_train:
                shutil.move(train_video, train_label_dir)
            # Test 데이터 이동
            test_label_dir = os.path.join(test_dir, label)
            os.makedirs(test_label_dir, exist_ok=True)
            for test_video in X_test:
                shutil.move(test_video, test_label_dir)

# 예시: 데이터셋을 80:20으로 나누어 디렉토리에 저장하기
data_dir = "C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\classified_video_320"
train_dir = "C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\real\\train"
test_dir = "C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\real\\test"

split_data_by_label(data_dir, train_dir, test_dir, test_size=0.2, random_state=42)

#####################
import os
import shutil
from sklearn.model_selection import train_test_split

def split_data_by_label(data_dir, train_dir, test_dir, test_size=0.2):
    # data_dir: 라벨별로 분류된 동영상이 있는 최상위 디렉토리
    # train_dir: Train 데이터를 저장할 디렉토리
    # test_dir: Test 데이터를 저장할 디렉토리

    # 모든 라벨 디렉토리 가져오기
    labels = os.listdir(data_dir)

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        train_label_dir = os.path.join(train_dir, label)
        test_label_dir = os.path.join(test_dir, label)

        # 라벨 내의 파일 목록 가져오기
        files = os.listdir(label_dir)

        # 파일을 셔플하여 나누기
        train_files, test_files = train_test_split(files, test_size=test_size, shuffle=True)

        # Train 디렉토리에 복사
        os.makedirs(train_label_dir, exist_ok=True)
        for file in train_files:
            src_path = os.path.join(label_dir, file)
            dest_path = os.path.join(train_label_dir, file)
            shutil.copy(src_path, dest_path)

        # Test 디렉토리에 복사
        os.makedirs(test_label_dir, exist_ok=True)
        for file in test_files:
            src_path = os.path.join(label_dir, file)
            dest_path = os.path.join(test_label_dir, file)
            shutil.copy(src_path, dest_path)

# 사용 예시
data_dir = "C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\classified_video_174"
train_dir = "C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\real\\train"
test_dir = "C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\real\\test"

split_data_by_label(data_dir, train_dir, test_dir)
