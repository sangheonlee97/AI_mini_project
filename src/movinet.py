import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

def list_files_per_class(zip_url):
  """
    List the files in each class of the dataset given the zip URL.

    Args:
      zip_url: URL from which the files can be unzipped. 

    Return:
      files: List of files in each of the classes.
  """
  files = []
  with rz.RemoteZip(URL) as zip:
    for zip_info in zip.infolist():
      files.append(zip_info.filename)
  return files

def get_class(fname):
  """
    Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the UCF101 dataset.

    Return:
      Class that the file belongs to.
  """
  return fname.split('_')[-3]

def get_files_per_class(files):
  """
    Retrieve the files that belong to each class. 

    Args:
      files: List of files in the dataset.

    Return:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  return files_for_class

def download_from_zip(zip_url, to_dir, file_names):
  """
    Download the contents of the zip file from the zip URL.

    Args:
      zip_url: Zip URL containing data.
      to_dir: Directory to download data to.
      file_names: Names of files to download.
  """
  with rz.RemoteZip(zip_url) as zip:
    for fn in tqdm.tqdm(file_names):
      class_name = get_class(fn)
      zip.extract(fn, str(to_dir / class_name))
      unzipped_file = to_dir / class_name / fn

      fn = pathlib.Path(fn).parts[-1]
      output_file = to_dir / class_name / fn
      unzipped_file.rename(output_file,)

def split_class_lists(files_for_class, count):
  """
    Returns the list of files belonging to a subset of data as well as the remainder of
    files that need to be downloaded.

    Args:
      files_for_class: Files belonging to a particular class of data.
      count: Number of files to download.

    Return:
      split_files: Files belonging to the subset of data.
      remainder: Dictionary of the remainder of files that need to be downloaded.
  """
  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

def download_ufc_101_subset(zip_url, num_classes, splits, download_dir):
  """
    Download a subset of the UFC101 dataset and split them into various parts, such as
    training, validation, and test. 

    Args:
      zip_url: Zip URL containing data.
      num_classes: Number of labels.
      splits: Dictionary specifying the training, validation, test, etc. (key) division of data 
              (value is number of files per split).
      download_dir: Directory to download data to.

    Return:
      dir: Posix path of the resulting directories containing the splits of data.
  """
  files = list_files_per_class(zip_url)
  for f in files:
    tokens = f.split('/')
    if len(tokens) <= 2:
      files.remove(f) # Remove that item from the list if it does not have a filename

  files_for_class = get_files_per_class(files)

  classes = list(files_for_class.keys())[:num_classes]

  for cls in classes:
    new_files_for_class = files_for_class[cls]
    random.shuffle(new_files_for_class)
    files_for_class[cls] = new_files_for_class

  # Only use the number of classes you want in the dictionary
  files_for_class = {x: files_for_class[x] for x in list(files_for_class)[:num_classes]}

  dirs = {}
  for split_name, split_count in splits.items():
    print(split_name, ":")
    split_dir = download_dir / split_name
    split_files, files_for_class = split_class_lists(files_for_class, split_count)
    download_from_zip(zip_url, split_dir, split_files)
    dirs[split_name] = split_dir

  return dirs

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result

class FrameGenerator:
  def __init__(self, path, n_frames, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.mp4'))
    classes = [p.parent.name for p in video_paths] 
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames) 
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label
      
URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
# download_dir = pathlib.Path('./UCF101_subset/')
# subset_paths = download_ufc_101_subset(URL, 
#                         num_classes = 10, 
#                         splits = {"train": 30, "test": 20}, 
#                         download_dir = download_dir)

train_path = pathlib.Path("C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\real\\train")
test_path = pathlib.Path("C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\real\\test")
# val_ds = pathlib.Path("C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\download_dir\\val")
subset_paths = { 'train' :  train_path,
                'test' :  test_path,
                # 'val' : val_ds
                }

batch_size = 5
num_frames = 200
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))
# print("dllfkjasdfl: ",subset_paths['train'])
train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training = False),
                                          output_signature = output_signature)
train_ds = train_ds.batch(batch_size)

# val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], num_frames, training = False),
#                                           output_signature = output_signature)
# val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames),
                                         output_signature = output_signature)
test_ds = test_ds.batch(batch_size)

# for frames, labels in test_ds.take(10):
#   print("labels : ",labels)
import os  
train_CLASSES = sorted(os.listdir("C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\real\\train"))
test_CLASSES = sorted(os.listdir("C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\real\\test"))

train_class_mapping = {i: class_name for i, class_name in enumerate(train_CLASSES)}
test_class_mapping = {i: class_name for i, class_name in enumerate(test_CLASSES)}

print(test_class_mapping)

model_id = 'a0'
resolution = 174

tf.keras.backend.clear_session()

backbone = movinet.Movinet(model_id=model_id)
backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

# Load pre-trained weights
# !wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_base.tar.gz -O movinet_a0_base.tar.gz -q
# !tar -xvf movinet_a0_base.tar.gz

# checkpoint_dir = f'movinet_{model_id}_base'
# checkpoint_dir = 'C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\w\\movinet_a0_base'
# checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
# checkpoint = tf.train.Checkpoint(model=model)
# status = checkpoint.restore(checkpoint_path)
# status.assert_existing_objects_matched()

def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model

model = build_classifier(batch_size, num_frames, resolution, backbone, 419)

num_epochs = 10

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

# 모델에 가중치 로드
model.load_weights(("../movinet_real_w3.h5"))

# results = model.fit(train_ds,
#                     # validation_data=val_ds,
#                     epochs=num_epochs,
#                     # validation_freq=1,
#                     verbose=1)



# res = model.evaluate(test_ds, return_dict=True)
# print("결과 !!!!!!!!!!!!!!!!!!!!")
# print(res)
# print("결과 !!!!!!!!!!!!!!!!!!!!")

# print("가중치 저장!!!!")
# model.save_weights("../movinet_real_w3.h5")
# model.save("../movinet_real_m3.h5")
# print("가중치 저장!!!!")
def get_actual_predicted_labels(dataset):
  """f
    Create a list of actual ground truth values and the predictions from the model.

    Args:
      dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

    Return:
      Ground truth and predicted values for a particular dataset.
  """
  actual = [labels for _, labels in dataset.unbatch()]
  predicted = model.predict(dataset)

  actual = tf.stack(actual, axis=0)
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return actual, predicted


  
fg = FrameGenerator(subset_paths['train'], num_frames, training = True)
label_names = list(fg.class_ids_for_name.keys())

actual, predicted = get_actual_predicted_labels(test_ds)

def return_real_word(y_pred, map):      # 폴더 순서대로 분류돼있던 클래스를  디렉토리 이름으로 변환해주는 함수
    temp = []
    for v in y_pred:
        temp.append(map[int(v)])
    return np.array(temp)
predict_num = predicted.numpy()
actual_num = actual.numpy()

y_pred = return_real_word(predict_num, train_class_mapping)
y_test = return_real_word(actual_num, test_class_mapping)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

import json
with open('token_dic.json', 'r', encoding = 'utf-8') as json_file:          ### 딕셔너리 불러오기
    token_dic = json.load(json_file)
flipped_dict = {v: k for k, v in token_dic.items()}

y_pred = return_real_word(y_pred, flipped_dict)
y_test = return_real_word(y_test, flipped_dict)

print("실제 데이터 : ", y_test)
print("예측 데이터 : ", y_pred)
print("acc : ", acc)

#==============================
from PIL import ImageFont, ImageDraw, Image
from pathlib import Path
import numpy as np
import cv2


# def get_label_for_video(y_test, y_pred, label_source):
#     actual_label = label_source[y_test]
#     predicted_label = label_source[y_pred]
#     return actual_label, predicted_label



def draw_text_on_frame(frame, text, position, font_path, font_size, font_color):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=font_color)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)


def find_first_video_files_and_labels(folder_path):
    video_paths = []
    base_path = Path(folder_path)
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            video_files = list(subfolder.glob('*.mp4'))
            if video_files:
                video_paths.append(str(video_files[0]))  # 첫 번째 비디오 파일 추가
                
                
    return video_paths
# 비디오 파일과 라벨을 함께 추출
video_paths = find_first_video_files_and_labels(test_path)
predicted_labels = y_pred.copy()


    
# 라벨 매핑 파일 로드


# font_path = "C:\\Users\\user\\Downloads\\nanum-all\\나눔 글꼴\\나눔고딕에코\\NanumFontSetup_TTF_GOTHICECO\\NanumGothicEco.ttf"
font_path = "C:\\Users\\AIA\\Documents\\카카오톡 받은 파일\\NanumGothicEco.ttf"



def play_videos_with_korean_labels(video_paths, actual_labels, predicted_labels, font_path):
    for index, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue

        # 리스트의 인덱스를 사용하여 각 비디오에 맞는 라벨 찾기
        actual_label = actual_labels[index]
        predicted_label = predicted_labels[index]
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                label_text = f"실제: {actual_label}, 예측: {predicted_label}"
                frame = draw_text_on_frame(frame, label_text, (10, 10), font_path, 15, (255, 255, 255))
                cv2.imshow('Video', frame)
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
                
        cap.release()
    cv2.destroyAllWindows()

# 영상 재생 및 라벨 표시
play_videos_with_korean_labels(video_paths, y_test, y_pred, font_path)
