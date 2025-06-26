# data_manager/folder_data.py
"""
FolderData 클래스
- 주어진 폴더에서 이미지 파일 목록을 수집하고, 각 파일을 FileData 객체로 관리
- 작업 대상 파일(work_file)을 설정
"""
import glob
import os
from .file_data import FileData


class FolderData:
    def __init__(self, path: str):
        self.__folder = path
        self.__files = []
        self.__work_file = None
        self.__init_work_folder()

    def __init_work_folder(self):
        FILE_EXT = ["png", "jpg", "gif"]
        target_files = []
        [
            target_files.extend(glob.glob(self.__folder + "/" + "*." + e))
            for e in FILE_EXT
        ]
        self.__files = [FileData(f) for f in target_files]

        if self.__files:
            self.__work_file = self.__files[0]

    def get_work_file(self):
        """현재 작업 대상 파일 반환"""
        return self.__work_file

    def set_work_file(self, target_file):
        """작업 대상 파일 변경"""
        self.__work_file = target_file

    def get_files(self):
        """모든 FileData 리스트 반환"""
        return self.__files

    def get_file_by_index(self, index):
        """지정 인덱스의 FileData 반환"""
        return self.__files[index]

    def get_files_as_string(self):
        """파일 이름 리스트 반환"""
        return [f.get_file_name() for f in self.__files]

    def get_folder_path(self):
        """작업 폴더 경로 반환"""
        return self.__folder
