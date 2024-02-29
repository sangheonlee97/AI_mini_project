# import sys
# sys.path.append("C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\src\\models-2.15.0")  # "path/to/official"을 실제로 다운로드한 "official" 디렉토리의 경로로 변경해야 합니다.
# C:\Users\AIA\Desktop\ai\AI_mini_project\src\models-2.15.0
import tqdm
import random
import pathlib
import itertools
import collections

# import cv2
import numpy as np
# import remotezip as rz
# import seaborn as sns
import matplotlib.pyplot as plt

# import keras
# import tensorflow as tf
# import tensorflow_hub as hub

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
