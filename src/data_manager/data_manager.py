# data_manager/data_manager.py
"""
DataManager 클래스
- 폴더 및 이미지 데이터 초기화
- OCR 처리 수행
- 출력 이미지 저장 및 탐색
"""
import os
import shutil
import easyocr
from tkinter import messagebox as mb

from .folder_data import FolderData

class DataManager:
    folder_data = None
    easyocr_reader = None

    @classmethod
    def init(cls):
        """기본 이미지 폴더 초기화 및 easyocr 모델 로딩"""
        default_path = os.path.join(os.getcwd(), "image")
        cls.reset_work_folder(default_path)
        cls.easyocr_reader = easyocr.Reader(['ch_sim', 'en'])

    @classmethod
    def reset_work_folder(cls, target_folder='./image'):
        """이미지 작업 폴더 재설정"""
        print('[DataManager.reset] target=', target_folder)
        target_path = os.path.abspath(target_folder)
        cls.folder_data = FolderData(target_path)
        cls.__init_output_folder(target_path)

    @classmethod
    def __init_output_folder(cls, target_folder):
        """출력 폴더 생성 및 이미지 복사"""
        output_folder = os.path.join(target_folder, '__OUTPUT_FILES__')
        os.makedirs(output_folder, exist_ok=True)
        for f in cls.folder_data.get_files():
            out_path = os.path.join(output_folder, os.path.basename(f.get_file_name()))
            if not os.path.exists(out_path):
                shutil.copy(f.get_file_name(), out_path)

    @classmethod
    def get_work_file(cls): return cls.folder_data.get_work_file()
    @classmethod
    def set_work_file(cls, file): cls.folder_data.set_work_file(file)

    @classmethod
    def get_output_file(cls):
        """출력 파일 경로 반환"""
        base = cls.folder_data.get_folder_path()
        name = os.path.basename(cls.folder_data.get_work_file().get_file_name())
        return os.path.join(base, '__OUTPUT_FILES__', name)

    @classmethod
    def save_output_file(cls, out_image):
        """이미지 결과 파일 저장"""
        out_file = cls.get_output_file()
        if out_image is None:
            print('이미지 없음, 저장 불가')
            return False
        if out_file.lower().endswith('png'):
            out_image.save(out_file)
        else:
            out_image.convert("RGB").save(out_file)
        print('이미지 저장 완료:', out_file)
        return True

    @classmethod
    def get_image_index(cls):
        """현재 작업 이미지의 인덱스 반환"""
        current = cls.folder_data.get_work_file()
        files = cls.folder_data.get_files()
        for i, f in enumerate(files):
            if f == current:
                return i
        return -1

    @classmethod
    def get_texts_from_image(cls):
        """
        OCR 수행 및 결과 텍스트 리스트 반환
        이미 OCR 수행된 경우 캐시된 텍스트 반환
        """
        index = cls.get_image_index()
        file = cls.folder_data.get_file_by_index(index)
        if file.is_ocr_executed():
            return file.get_texts_as_string()

        ocr_result = cls.easyocr_reader.readtext(file.get_file_name())
        file.set_texts(ocr_result)
        texts = file.get_texts_as_string()
        if not texts:
            mb.showwarning("경고", "text를 찾을 수 없습니다.")
            return None
        return texts
