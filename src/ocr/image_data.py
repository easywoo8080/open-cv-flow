import cv2
import numpy as np

class FaceData:
    def __init__(self, name, position):
        self.name = name
        self.position = position  # (x, y, w, h)

class TextData:
    def __init__(self, text, position):
        self.text = text
        self.position = position

class ImageData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.texts = []
        self.faces = []

    def set_faces(self, positions):
        if positions is not None and isinstance(positions, np.ndarray) and len(positions) > 0:
            self.faces = [FaceData(f'얼굴-{i+1:04d}', p) for i, p in enumerate(positions)]

    def detect_faces(self):
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(self.file_path)
        if img is None:
            raise FileNotFoundError(f'이미지 없음: {self.file_path}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 4)
        self.set_faces(faces)
        return img  # 이미지도 같이 반환
