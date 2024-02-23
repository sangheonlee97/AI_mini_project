import os

def remove_padding_zeros(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.mp4'):
            # 파일 이름에서 확장자 제거
            name_without_extension = os.path.splitext(filename)[0]

            # 숫자 부분의 패딩된 0 제거
            stripped_name = str(int(name_without_extension))

            # 새로운 파일 이름 생성
            new_filename = stripped_name + '.mp4'

            # 경로 변경
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)

            # 이름 변경
            os.rename(old_path, new_path)

# 디렉토리 경로 설정
video_directory = '../resource/video/'

# 함수 호출
remove_padding_zeros(video_directory)