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
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.faces = []

    def detect_faces_cv2(self):
        """
        OpenCV의 Haar Cascade 분류기를 사용해 이미지에서 얼굴을 검출한다.

        처리 과정:
        1. 정면 얼굴 검출용 Haar Cascade XML 파일을 로드한다.
        2. 지정된 파일 경로에서 이미지를 읽어들인다.
        3. 처리 속도와 정확도를 위해 이미지를 그레이스케일로 변환한다.
        4. 분류기를 사용해 얼굴 위치를 검출한다.
        - scaleFactor: 이미지 크기를 단계별로 축소하는 비율 (기본값 1.1)
        - minNeighbors: 얼굴 후보 사각형이 가져야 할 최소 인접 개수 (기본값 5)
        5. 얼굴이 검출되면, 각각에 고유한 이름을 부여하고 위치 정보와 함께 저장한다.

        업데이트되는 속성:
            self.faces (List[FaceData]): 검출된 얼굴의 이름과 위치 정보를 담은 리스트
        """
        # 1. OpenCV의 Haar Cascade 분류기 불러오기
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # => 얼굴 검출에 사용되는 사전 학습된 XML 파일을 로드한다.

        # 2. 이미지 파일 읽기
        image = cv2.imread(self.file_path)
        # => self.file_path 경로의 이미지를 BGR 컬러로 읽어들인다.

        # 3. 그레이스케일로 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # => 얼굴 검출은 색상 정보 불필요하므로, 처리 속도와 정확도 향상을 위해 흑백 이미지로 변환한다.

        # 4. 얼굴 위치 검출
        positions = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # => 이미지 내 얼굴 위치를 (x, y, w, h) 형태로 배열로 반환한다.
        #    scaleFactor: 이미지 크기 조정 비율 (1.1은 10%씩 축소하며 탐색)
        #    minNeighbors: 얼굴로 인식할 최소 인접 사각형 수(클러스터 크기)

        # 5. 검출된 얼굴 정보 저장
        if len(positions) > 0:
            self.faces = [FaceData(f'얼굴-{i+1:04d}', pos) for i, pos in enumerate(positions)]
        # => 검출된 얼굴마다 고유 이름과 위치 정보를 FaceData 객체로 생성해 리스트에 저장한다.

