import os

directory_path = '../resource/수어/영상/'

for filename in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, filename)):
        # 파일의 확장자를 제외한 이름을 추출
        name, ext = os.path.splitext(filename)
        
        # 파일 이름의 뒤에서 5자리 숫자 추출
        new_name = name[-5:]
        
        # 새로운 파일 이름으로 변경
        new_filename = os.path.join(directory_path, new_name + ext)
        
        # 파일 이름 변경
        os.rename(os.path.join(directory_path, filename), new_filename)
