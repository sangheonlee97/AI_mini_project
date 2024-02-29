import os
import pandas as pd
import shutil

def classify_videos(input_dir, output_dir, map_csv):
    for filename in os.listdir(input_dir):
        if filename.endswith('.mp4'):
            video_name = os.path.splitext(filename)[0]
            s = map_csv[map_csv['번호'] == int(video_name)]['한국어']
            video_word = str(s.iloc[0])
            
            # Create destination directory if it doesn't exist
            dest_dir = os.path.join(output_dir, video_word)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Move the video file
            src_path = os.path.join(input_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.move(src_path, dest_path)


# Example usage:
csv_path = 'C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\tokend_word.csv'
video_dir = 'C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\video_standardized_320320\\'
output_dir = 'C:\\Users\\AIA\\Desktop\\ai\\AI_mini_project\\resource\\classified_video_320\\'
df = pd.read_csv(csv_path)

classify_videos(video_dir, output_dir, df)
