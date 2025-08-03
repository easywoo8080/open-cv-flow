def draw_weight_info_with_outline_box(frame, landmarks, total_weight=100.0):
    weights = {}
    positions = {}

    for name, idx in landmark_ids.items():
        lm = landmarks[idx]
        raw_pos = (int(lm.x * width), int(lm.y * height))
        prev = prev_positions.get(name)
        smoothed_pos = smooth_position(prev, raw_pos)
        prev_positions[name] = smoothed_pos

        positions[name] = smoothed_pos
        weights[name] = total_weight * weight_ratios[name]

    # 사람 전체를 감싸는 bounding box 계산
    x_vals = [p[0] for p in positions.values()]
    y_vals = [p[1] for p in positions.values()]
    x_min, x_max = min(x_vals) - 20, max(x_vals) + 20
    y_min, y_max = min(y_vals) - 30, max(y_vals) + 30

    # 사각형 그리기
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (100, 100, 100), 2)

    # 표준편차 기준
    values = np.array(list(weights.values()))
    mean, std = values.mean(), values.std()

    # 텍스트 분류
    top_labels = ["left_shoulder", "right_shoulder"]
    bottom_labels = ["left_knee", "right_knee", "left_heel", "right_heel", "left_hip", "right_hip"]

    # 상단 텍스트 배치
    top_spacing = (x_max - x_min) // (len(top_labels) + 1)
    for i, name in enumerate(top_labels):
        value = weights[name]
        color = (0, 0, 255) if abs(value - mean) > std else (0, 255, 0)
        text = f"{name}:{value:.1f}kg"
        x = x_min + top_spacing * (i + 1)
        y = y_min - 10  # 위쪽 테두리 밖
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 하단 텍스트 배치
    bottom_spacing = (x_max - x_min) // (len(bottom_labels) + 1)
    for i, name in enumerate(bottom_labels):
        value = weights[name]
        color = (0, 0, 255) if abs(value - mean) > std else (0, 255, 0)
        text = f"{name}:{value:.1f}kg"
        x = x_min + bottom_spacing * (i + 1)
        y = y_max + 20  # 아래쪽 테두리 밖
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
