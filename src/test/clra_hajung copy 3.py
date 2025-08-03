import cv2
import mediapipe as mp
import numpy as np
import random

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('sample.mp4')

# 하중 기록용 (moving average)
load_history = {
    'left_shoulder': [],
    'right_shoulder': [],
    'left_foot': [],
    'right_foot': []
}

MAX_HISTORY = 30  # 표준편차 계산을 위한 기록 길이

def draw_text_border(image, text, org, font, scale, color, thickness):
    cv2.putText(image, text, org, font, scale, (0,0,0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, font, scale, color, thickness, cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        # 좌표 추출
        def get_point(part):
            lm = landmarks[mp_pose.PoseLandmark[part].value]
            return int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])

        # 관심 부위
        joints = {
            'left_shoulder': get_point('LEFT_SHOULDER'),
            'right_shoulder': get_point('RIGHT_SHOULDER'),
            'left_foot': get_point('LEFT_ANKLE'),
            'right_foot': get_point('RIGHT_ANKLE')
        }

        # 인물 전체 bounding box
        all_x = [lm.x * frame.shape[1] for lm in landmarks]
        all_y = [lm.y * frame.shape[0] for lm in landmarks]
        min_x, max_x = int(min(all_x)), int(max(all_x))
        min_y, max_y = int(min(all_y)), int(max(all_y))
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        width = int((max_x - min_x) * 2.0)
        height = int((max_y - min_y) * 2.0)

        top_left = (center_x - width // 2, center_y - height // 2)
        bottom_right = (center_x + width // 2, center_y + height // 2)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)

        # 각 부위별 하중 표시
        for joint_name, coord in joints.items():
            # 가짜 하중값 갱신
            load = random.uniform(20, 100)
            load_history[joint_name].append(load)
            if len(load_history[joint_name]) > MAX_HISTORY:
                load_history[joint_name].pop(0)

            # 표준편차 이상 여부 판별
            history = np.array(load_history[joint_name])
            std_dev = np.std(history)
            avg = np.mean(history)
            is_abnormal = abs(load - avg) > std_dev * 1.5

            color = (0, 0, 255) if is_abnormal else (0, 255, 0)
            load_str = f"{joint_name.replace('_', ' ').title()}: {load:.1f}kg"

            # 텍스트 위치 계산 (사각형 테두리 기준)
            if 'shoulder' in joint_name:
                # 위쪽 테두리
                pos = (top_left[0] + 10 if 'left' in joint_name else bottom_right[0] - 250, top_left[1] + 25)
            elif 'foot' in joint_name:
                # 아래쪽 테두리
                pos = (top_left[0] + 10 if 'left' in joint_name else bottom_right[0] - 250, bottom_right[1] - 10)
            else:
                pos = coord

            draw_text_border(frame, load_str, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 포즈 그리기
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Load Monitor', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
