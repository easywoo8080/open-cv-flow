import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def angle_alert(angle, joint):
    if joint == "knee":
        return angle < 90 or angle > 180
    if joint == "hip":
        return angle < 70 or angle > 180
    if joint == "ankle":
        return angle < 70 or angle > 110
    if joint == "shoulder":
        return angle < 70 or angle > 180
    return False

cap = cv2.VideoCapture("sample.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        def get_point(name):
            lm = landmarks[name]
            return (int(lm.x * w), int(lm.y * h))

        # 주요 부위 좌표
        l_shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        r_shoulder = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        l_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP.value)
        r_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
        l_knee = get_point(mp_pose.PoseLandmark.LEFT_KNEE.value)
        r_knee = get_point(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        l_ankle = get_point(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        r_ankle = get_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        # 전체 바운딩 박스
        x_coords = [p[0] for p in [l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]]
        y_coords = [p[1] for p in [l_shoulder, r_shoulder, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        pad_w = int((x_max - x_min) * 0.25)
        pad_h = int((y_max - y_min) * 0.25)
        x_min -= pad_w
        x_max += pad_w
        y_min -= pad_h
        y_max += pad_h

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        def draw_text(text, x, y, is_alert):
            color = (0, 0, 255) if is_alert else (0, 255, 0)
            cv2.putText(frame, text, (x, y), font, 0.6, color, 2, cv2.LINE_AA)

        # 각도 계산
        left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        left_shoulder_angle = calculate_angle(l_hip, l_shoulder, r_shoulder)
        right_shoulder_angle = calculate_angle(r_hip, r_shoulder, l_shoulder)

        def ankle_angle(knee, ankle):
            fake_foot = (ankle[0], ankle[1] + 20)
            return calculate_angle(knee, ankle, fake_foot)

        left_ankle_angle = ankle_angle(l_knee, l_ankle)
        right_ankle_angle = ankle_angle(r_knee, r_ankle)

        # 텍스트 왼쪽, 오른쪽 정렬 위치
        left_x = x_min - 220
        right_x = x_max + 10

        # 텍스트 세로 시작 위치와 간격
        start_y = y_min - 20
        line_height = 30

        # 왼쪽 관절 텍스트 (위부터 아래)
        draw_text(f"L Shoulder Angle: {left_shoulder_angle:.1f}", left_x, start_y, angle_alert(left_shoulder_angle, "shoulder"))
        draw_text(f"L Hip Angle: {left_hip_angle:.1f}", left_x, start_y + line_height, angle_alert(left_hip_angle, "hip"))
        draw_text(f"L Knee Angle: {left_knee_angle:.1f}", left_x, start_y + 2 * line_height, angle_alert(left_knee_angle, "knee"))
        draw_text(f"L Ankle Angle: {left_ankle_angle:.1f}", left_x, start_y + 3 * line_height, angle_alert(left_ankle_angle, "ankle"))

        # 오른쪽 관절 텍스트 (위부터 아래)
        draw_text(f"R Shoulder Angle: {right_shoulder_angle:.1f}", right_x, start_y, angle_alert(right_shoulder_angle, "shoulder"))
        draw_text(f"R Hip Angle: {right_hip_angle:.1f}", right_x, start_y + line_height, angle_alert(right_hip_angle, "hip"))
        draw_text(f"R Knee Angle: {right_knee_angle:.1f}", right_x, start_y + 2 * line_height, angle_alert(right_knee_angle, "knee"))
        draw_text(f"R Ankle Angle: {right_ankle_angle:.1f}", right_x, start_y + 3 * line_height, angle_alert(right_ankle_angle, "ankle"))

        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Angle-based Load Monitor", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
