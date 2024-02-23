import cv2
import os
import numpy as np
import pandas as pd
import re
    
def extract_frames(input_dir, output_dir, target_frame_count, map_csv):
    for filename in os.listdir(input_dir):
        if filename.endswith('.mp4'):
            input_path = os.path.join(input_dir, filename)
            match = re.match(r'\d+', filename)
            if match:
                video_name = match.group()
                s = map_csv[map_csv['번호'] == int(video_name)]['한국어']
                print(s.values)
                video_word = str(s.iloc[0])
            else:
                continue

            output_subdir = os.path.join(output_dir, video_word)

            # Create a subdirectory if it doesn't exist
            os.makedirs(output_subdir, exist_ok=True)

            # Open video file
            cap = cv2.VideoCapture(input_path)

            # Determine target frame count for padding
            current_frame_count = int(cap.get(7))  # Get total number of frames
            remaining_frames = target_frame_count - current_frame_count
            padding_frames = max(0, remaining_frames)

            # Read the first frame to use for padding
            _, first_frame = cap.read()

            # Padding with the first frame if necessary
            for i in range(padding_frames):
                cv2.imwrite(f"{output_subdir}/{video_name}_frame{i}.png", first_frame)

            # Read and write frames
            for i in range(current_frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # Write the frame as an image
                cv2.imwrite(f"{output_subdir}/{video_name}_frame{i + padding_frames}.png", frame)

    # Release everything when done
    cv2.destroyAllWindows()

# Example usage
map_csv = pd.read_csv('../resource/tokend_word.csv')
input_video_dir = '../resource/video_standardized_7050/'
output_image_dir = '../resource/test_pad/test/'
# print(map_csv[map_csv['번호'] == 100]['한국어'])
extract_frames(input_video_dir, output_image_dir, target_frame_count=200+1, map_csv=map_csv)
