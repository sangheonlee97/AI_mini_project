import os
import subprocess
import pandas as pd

def standardize_resolution(input_dir, output_dir, target_resolution, csv):
    os.makedirs(output_dir, exist_ok=True)
    for filename in csv['번호']:
        if filename.endswith('.mp4'):
            print("번호 시작", filename)
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.mp4'
            output_path = os.path.join(output_dir, output_filename)

            subprocess.run(['ffmpeg', '-i', input_path, '-vf', f'scale={target_resolution[0]}:{target_resolution[1]}', '-c:a', 'copy', output_path])
            print("번호 끝", filename)
            

# 디렉토리 경로 및 원하는 해상도 설정
input_directory = '..//resource//video'
output_directory = '..//resource//video_standardized_174'
target_resolution = (174, 174)
csv = pd.read_csv('../resource/tokend_word.csv')
csv['번호'] = csv['번호'].astype(str) + '.mp4'
print(csv)
# 함수 호출
standardize_resolution(input_directory, output_directory, target_resolution, csv)
