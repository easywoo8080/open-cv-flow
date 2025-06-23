# data_manager/text_data.py
"""
TextData 클래스
- OCR 텍스트 및 좌표 정보를 저장
- 번역된 텍스트(tr_text)와 원본 위치정보(position) 관리
"""
class TextData:
    def __init__(self, text, position):
        self.__text = text
        self.__position_info = position  # [[x1, y1], ..., [x4, y4]]
        self.__tr_text = None

    def set_tr_text_with_position(self, tr_text, position):
        """번역 텍스트 및 위치 정보 설정"""
        self.__tr_text = tr_text
        self.__position_info = position

    def get_text(self): return self.__text
    def get_position_info(self): return self.__position_info
    def get_tr_text(self): return self.__tr_text
