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

class FaceDetection:
    def __init__(self, file_path):
        self.file_path = file_path
        self.texts = []
        self.faces = []

    def set_faces(self, positions):
        """
        Set face data from detected face positions.

        Args:
            positions (np.ndarray): 배열 형태의 얼굴 위치 정보 (x, y, w, h).
        """
        if positions is not None and isinstance(positions, np.ndarray) and len(positions) > 0:
            self.faces = [FaceData(f'얼굴-{i+1:04d}', p) for i, p in enumerate(positions)]

    def detect_faces(self):
        """
        Detect faces in the image file specified by `file_path`.

        Returns:
            np.ndarray: 원본 이미지 (BGR).
        
        Raises:
            FileNotFoundError: 이미지 파일을 찾지 못했을 경우.
        """
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(self.file_path)
        if img is None:
            raise FileNotFoundError(f'이미지 없음: {self.file_path}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 4)
        self.set_faces(faces)
        return img

    def remove_alpha(self, img: np.ndarray) -> np.ndarray:
        """
        이미지에서 알파 채널 제거하고 흰 배경으로 합성한 RGB 이미지 반환
        Args:
            img (np.ndarray): RGBA 또는 RGB 이미지 배열
        Returns:
            np.ndarray: 알파 채널 제거된 RGB 이미지
        """
        if img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            rgb = img[:, :, :3].astype(float)
            white_bg = np.ones_like(rgb) * 255
            img_rgb = (rgb * alpha[:, :, None] + white_bg * (1 - alpha[:, :, None])).astype(np.uint8)
        else:
            img_rgb = img
        return img_rgb

    def binarize_image(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Convert an RGB image to a binarized grayscale image using Otsu's thresholding.

        Args:
            img_rgb (np.ndarray): RGB 이미지 (BGR 형식, dtype=uint8).

        Returns:
            np.ndarray: 이진화된 흑백 이미지 (dtype=uint8).
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

