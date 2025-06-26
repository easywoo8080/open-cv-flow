# data_manager/face_data.py
"""
FaceData 클래스
- 감지된 얼굴 객체의 이름 및 위치 정보 저장
"""


class FaceData:
    def __init__(self, name, position):
        self.__name = name
        self.__position = position

    def get_name(self):
        return self.__name

    def get_position_info(self):
        return self.__position

    def __str__(self):
        return self.__name
