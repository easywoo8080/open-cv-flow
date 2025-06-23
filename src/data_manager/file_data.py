# data_manager/file_data.py
"""
FileData 클래스
- 이미지 파일 1개에 대한 정보 저장 (OCR 결과, 얼굴 인식 결과 등)
- OCR 처리 여부 및 텍스트, 얼굴 데이터 관리
"""
import numpy
from .text_data import TextData
from .face_data import FaceData

class FileData:
    def __init__(self, file):
        self.__name = file
        self.__texts = []
        self.__is_ocr_executed = False
        self.__faces = None

    def get_file_name(self): return self.__name
    def clear_faces(self): self.__faces = None
    def get_faces(self): return self.__faces

    def get_faces_as_string(self):
        """감지된 얼굴 이름 리스트 반환"""
        if not self.__faces: return None
        return [f.get_name() for f in self.__faces]

    def set_faces(self, positions):
        """
        얼굴 위치 배열을 받아 FaceData 리스트 생성
        param positions: numpy.ndarray
        """
        if isinstance(positions, numpy.ndarray) and len(positions) > 0:
            self.__faces = [FaceData(f'얼굴-{i+1:04d}', p) for i, p in enumerate(positions)]

    def is_ocr_executed(self): return self.__is_ocr_executed

    def set_texts(self, ocr_texts):
        """
        OCR 결과 리스트를 받아 TextData 리스트로 저장
        param ocr_texts: List[Tuple[polygon, text, conf]]
        """
        self.__texts = [TextData(t[1], t[0]) for t in ocr_texts]
        self.__is_ocr_executed = True

    def get_texts(self): return self.__texts
    def get_text_by_index(self, index): return self.__texts[index]
    def get_texts_as_string(self): return [t.get_text() for t in self.__texts]
    def get_positions_as_string(self): return [t.get_position_info() for t in self.__texts]
    def get_text_as_string_by_index(self, index): return self.get_texts_as_string()[index]

    def get_rectangle_position_by_texts_index(self, index):
        """
        OCR 텍스트 위치에서 좌측상단과 우측하단 좌표 반환
        return: (start_pos, end_pos)
        """
        rect = self.get_text_by_index(index).get_position_info()
        return rect[0], rect[2]
