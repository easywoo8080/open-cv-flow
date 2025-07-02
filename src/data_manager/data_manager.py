"""
DataManager 모듈
================

이미지 폴더를 초기화하고 EasyOCR로 OCR 처리를 수행하며,
결과 이미지를 저장·탐색하는 기능을 제공한다.
"""

from __future__ import annotations

import os
import shutil
from tkinter import messagebox as mb

import easyocr

from .folder_data import FolderData


class DataManager:
    """이미지 데이터 및 OCR 처리를 관리하는 클래스.

    Attributes:
        folder_data: 현재 작업 폴더 및 이미지 메타 정보를 담는 객체.
        easyocr_reader: EasyOCR 모델 인스턴스.
    """

    folder_data: FolderData | None = None
    easyocr_reader: easyocr.Reader | None = None

    # --------------------------------------------------------------------- #
    # 기본 유틸리티                                                          #
    # --------------------------------------------------------------------- #
    @staticmethod
    def file_exists(path: str) -> bool:
        """주어진 경로의 파일 존재 여부를 확인한다.

        Args:
            path: 확인할 파일 경로.

        Returns:
            bool: 파일이 존재하면 ``True``, 그렇지 않으면 ``False``.
        """
        return os.path.exists(path)

    # --------------------------------------------------------------------- #
    # 초기화 단계                                                            #
    # --------------------------------------------------------------------- #
    @classmethod
    def init(cls) -> None:
        """기본 이미지 폴더 및 EasyOCR 모델을 초기화한다.

        - 현재 작업 디렉터리 하위의 ``image`` 폴더를 작업 폴더로 설정한다.
        - EasyOCR 리더를 한국어·영어 모델로 로딩한다.
        """
        default_path = os.path.join(os.getcwd(), "image")
        cls.reset_work_folder(default_path)
        cls.easyocr_reader = easyocr.Reader(["ch_sim", "en"])

    @classmethod
    def reset_work_folder(cls, target_folder: str = "./image") -> None:
        """작업 폴더를 재설정한다.

        Args:
            target_folder: 새 작업 폴더 경로. 기본값은 ``./image``.

        Side Effects:
            - :pyattr:`folder_data`를 새로 생성한다.
            - 출력 폴더( ``__OUTPUT_FILES__`` )를 초기화한다.
        """
        print("[DataManager.reset] target=", target_folder)
        target_path = os.path.abspath(target_folder)
        cls.folder_data = FolderData(target_path)
        cls.__init_output_folder(target_path)

    @classmethod
    def __init_output_folder(cls, target_folder: str) -> None:
        """출력 폴더를 생성하고 원본 이미지를 복사한다.

        Args:
            target_folder: 작업 폴더 경로.

        Notes:
            - 존재하지 않는 경우 ``__OUTPUT_FILES__`` 폴더를 생성한다.
            - 이미 복사된 파일은 덮어쓰지 않는다.
        """
        output_folder = os.path.join(target_folder, "__OUTPUT_FILES__")
        os.makedirs(output_folder, exist_ok=True)

        if cls.folder_data is not None:
            for f in cls.folder_data.get_files():
                out_path = os.path.join(
                    output_folder, os.path.basename(f.get_file_name())
                )
                if not os.path.exists(out_path):
                    shutil.copy(f.get_file_name(), out_path)

    # --------------------------------------------------------------------- #
    # 파일 접근                                                              #
    # --------------------------------------------------------------------- #
    @classmethod
    def get_work_file(cls):
        """현재 작업 중인 파일 객체를 반환한다.

        Returns:
            FolderFile | None: 작업 파일 객체. 설정되지 않았으면 ``None``.
        """
        if cls.folder_data is not None:
            return cls.folder_data.get_work_file()
        return None

    @classmethod
    def set_work_file(cls, file) -> None:
        """작업 파일을 설정한다.

        Args:
            file: :pyclass:`FolderFile` 인스턴스.
        """
        if cls.folder_data is not None:
            cls.folder_data.set_work_file(file)

    # --------------------------------------------------------------------- #
    # 경로 계산                                                              #
    # --------------------------------------------------------------------- #
    @classmethod
    def get_output_file(cls) -> str:
        """현재 작업 파일의 출력 경로를 계산한다.

        Returns:
            str: ``<작업폴더>/__OUTPUT_FILES__/<파일명>`` 형식의 경로.

        Raises:
            RuntimeError: ``folder_data`` 또는 ``work_file`` 이 설정되지 않았을 때.
        """
        if cls.folder_data is None or cls.folder_data.get_work_file() is None:
            raise RuntimeError("folder_data 또는 work_file이 설정되지 않았습니다.")

        base = cls.folder_data.get_folder_path()
        name = os.path.basename(cls.folder_data.get_work_file().get_file_name())
        return os.path.join(base, "__OUTPUT_FILES__", name)

    # --------------------------------------------------------------------- #
    # 결과 저장                                                              #
    # --------------------------------------------------------------------- #
    @classmethod
    def save_output_file(cls, out_image) -> bool:
        """OCR 결과 이미지를 디스크에 저장한다.

        Args:
            out_image: PIL 이미지 객체.

        Returns:
            bool: 저장 성공 여부.
        """
        out_file = cls.get_output_file()
        if out_image is None:
            print("이미지 없음, 저장 불가")
            return False

        if out_file.lower().endswith("png"):
            out_image.save(out_file)
        else:
            out_image.convert("RGB").save(out_file)

        print("이미지 저장 완료:", out_file)
        return True

    # --------------------------------------------------------------------- #
    # 인덱스 및 OCR                                                         #
    # --------------------------------------------------------------------- #
    @classmethod
    def get_image_index(cls) -> int:
        """현재 작업 이미지의 인덱스를 반환한다.

        Returns:
            int: 0 이상의 인덱스, 찾지 못하면 ``-1``.
        """
        current = cls.folder_data.get_work_file()
        files = cls.folder_data.get_files()
        for i, f in enumerate(files):
            if f == current:
                return i
        return -1

    @classmethod
    def get_texts_from_image(cls):
        """현재 작업 이미지에서 OCR을 수행해 텍스트를 반환한다.

        - 이미 OCR 결과가 캐시돼 있으면 캐시를 사용한다.
        - 텍스트가 없으면 경고 메시지를 띄운다.

        Returns:
            list[str] | None: 텍스트 문자열 리스트. 실패 시 ``None``.

        Side Effects:
            - 필요 시 EasyOCR를 실행해 캐시를 갱신한다.
            - 오류·경고 상황에서 Tkinter 메시지박스를 표시한다.
        """
        index = cls.get_image_index()
        file = (
            cls.folder_data.get_file_by_index(index)
            if cls.folder_data is not None
            else None
        )

        if file is None:
            mb.showwarning("경고", "파일을 찾을 수 없습니다.")
            return None

        if file.is_ocr_executed():
            return file.get_texts_as_string()

        # OCR 수행
        ocr_result = cls.easyocr_reader.readtext(file.get_file_name())
        file.set_texts(ocr_result)
        texts = file.get_texts_as_string()

        if not texts:
            mb.showwarning("경고", "text를 찾을 수 없습니다.")
            return None

        return texts
