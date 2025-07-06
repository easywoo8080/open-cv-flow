from src.files.folder_manager import FolderManager
from src.ocr.image_data import ImageData

class DataManager:
    def __init__(self, folder_path):
        self.folder_manager = FolderManager(folder_path)
        self.image_data_map = {f: ImageData(f) for f in self.folder_manager.get_files()}
    # ...OCR 등 기능 추가...