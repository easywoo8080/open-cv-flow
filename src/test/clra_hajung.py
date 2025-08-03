import cv2
import mediapipe as mp

# 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 동영상 입력
cap = cv2.VideoCapture('sample.mp4')

# 동영상 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 지정 (mp4)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# 하중 추정 함수
def estimate_weight_bias(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
    right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]

    pelvis_center_x = (left_hip.x + right_hip.x) / 2
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    body_center_x = (pelvis_center_x + shoulder_center_x) / 2
    foot_center_x = (left_heel.x + right_heel.x) / 2

    bias = body_center_x - foot_center_x
    return bias

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

        # 하중 추정 및 표시
        bias = estimate_weight_bias(result.pose_landmarks.landmark)
        if bias > 0.03:
            weight_text = "Right side load ↑"
        elif bias < -0.03:
            weight_text = "Left side load ↑"
        else:
            weight_text = "Balanced load"

        cv2.putText(frame, weight_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
out.release()
cv2.destroyAllWindows()
