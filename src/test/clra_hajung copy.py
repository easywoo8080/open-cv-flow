import cv2
import mediapipe as mp
import numpy as np

# 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture('sample.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# 하중 비중
weight_ratios = {
    "left_shoulder": 0.1,
    "right_shoulder": 0.1,
    "left_hip": 0.2,
    "right_hip": 0.2,
    "left_knee": 0.15,
    "right_knee": 0.15,
    "left_heel": 0.05,
    "right_heel": 0.05,
}

landmark_ids = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_heel": mp_pose.PoseLandmark.LEFT_HEEL,
    "right_heel": mp_pose.PoseLandmark.RIGHT_HEEL,
}

# 이전 위치 저장용 (이동 스무딩)
prev_positions: dict[str, tuple[int, int]] = {}
smooth_factor = 0.6  # 0~1 사이 값: 1에 가까울수록 움직임 작음

def smooth_position(prev: tuple[int, int], new: tuple[int, int]) -> tuple[int, int]:
    if prev is None:
        return new
    x = int(prev[0] * smooth_factor + new[0] * (1 - smooth_factor))
    y = int(prev[1] * smooth_factor + new[1] * (1 - smooth_factor))
    return (x, y)

def draw_weight_info(frame, landmarks, total_weight=100.0):
    weights = {}
    positions = {}

    for name, idx in landmark_ids.items():
        lm = landmarks[idx]
        raw_pos = (int(lm.x * width), int(lm.y * height))

        # 위치 스무딩
        prev = prev_positions.get(name)
        smoothed_pos = smooth_position(prev, raw_pos)
        prev_positions[name] = smoothed_pos

        positions[name] = smoothed_pos
        weights[name] = total_weight * weight_ratios[name]

    # 통계
    values = np.array(list(weights.values()))
    mean, std = values.mean(), values.std()

    for name, value in weights.items():
        pos = positions[name]
        color = (0, 0, 255) if abs(value - mean) > std else (0, 255, 0)

        # 텍스트 박스 그리기
        text = f"{value:.1f}kg"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        box_pos = (pos[0] + 10, pos[1] - 20)

        # 사각형 박스
        cv2.rectangle(frame, (box_pos[0] - 4, box_pos[1] - th - 4),
                      (box_pos[0] + tw + 4, box_pos[1] + 4), color, -1)

        # 텍스트
        cv2.putText(frame, text, (box_pos[0], box_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 위치 점
        cv2.circle(frame, pos, 5, color, -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
        )

        draw_weight_info(frame, result.pose_landmarks.landmark)

    out.write(frame)
    cv2.imshow('Pose Load Estimation', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
