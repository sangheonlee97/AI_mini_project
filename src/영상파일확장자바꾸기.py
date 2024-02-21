import os
import subprocess

def change_extension(input_dir, old_ext, new_ext):
    for filename in os.listdir(input_dir):
        if filename.endswith(old_ext):
            old_path = os.path.join(input_dir, filename)
            new_filename = os.path.splitext(filename)[0] + '.' + new_ext
            new_path = os.path.join(input_dir, new_filename)

            # 파일 이름 변경
            os.rename(old_path, new_path)

            # 확장자 변경 (ffmpeg 이용)
            subprocess.run(['ffmpeg', '-i', new_path, new_path])

# 디렉토리 경로, 변경 전 확장자, 변경 후 확장자 설정
input_directory = '/path/to/your/directory'
old_extension = 'avi'  # 현재 사용 중인 확장자
new_extension = 'mp4'  # 변경할 확장자

# 함수 호출
change_extension(input_directory, old_extension, new_extension)