from __future__ import annotations
import cv2 as cv
import numpy as np
import os


# ------------------------------
# 1) 기준 프레임 로드
# ------------------------------
video_path = r"F:\git_hub\open-cv-flow\src\test\sample.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"영상 파일을 찾을 수 없음: {video_path}")

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("영상 열기 실패")

ok, frame0 = cap.read()
if not ok:
    raise RuntimeError("첫 프레임 읽기 실패")

h, w, c = frame0.shape
print("영상 크기:", w, "x", h)


# ------------------------------
# 2) 색 표본 좌표 선택 (마우스)
# ------------------------------
samples: list[tuple[int, int]] = []

def mouse_cb(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        samples.append((x, y))
        cv.circle(frame0, (x, y), 5, (0, 0, 255), -1)
        cv.imshow("pick samples", frame0)
        print(f"샘플 추가: {(x, y)}")

print("영상에서 같은 색 영역(홀드)을 클릭하세요. 끝내려면 아무 키나 누르세요.")
cv.imshow("pick samples", frame0)
cv.setMouseCallback("pick samples", mouse_cb)
cv.waitKey(0)
cv.destroyAllWindows()

if len(samples) == 0:
    raise RuntimeError("색 표본이 선택되지 않았습니다.")


# ------------------------------
# 3) 색상 분포(평균 + 공분산) 계산
# ------------------------------
lab = cv.cvtColor(frame0, cv.COLOR_BGR2LAB)
vals = np.array([lab[pt[1], pt[0]] for pt in samples], dtype=np.float32)
mu = vals.mean(axis=0)
cov = np.cov(vals.T) + np.eye(3) * 1.0  # 안정성 보정
inv_cov = np.linalg.inv(cov)


def maha_mask(img_bgr: np.ndarray, thresh: float = 6.0) -> np.ndarray:
    """Mahalanobis 거리 기반 색 분할 마스크"""
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB).astype(np.float32)
    diff = lab - mu
    d2 = (diff @ inv_cov * diff).sum(axis=2)  # Mahalanobis distance^2
    mask = (d2 < thresh**2).astype(np.uint8) * 255
    # 모폴로지 정제
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return mask


# ------------------------------
# 4) 전 프레임 누적 마스크
# ------------------------------
acc = None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    m = maha_mask(frame, thresh=6.0)
    acc = m if acc is None else cv.bitwise_or(acc, m)

cap.release()


# ------------------------------
# 5) 연결요소로 홀드 후보 추출
# ------------------------------
num, labels, stats, centroids = cv.connectedComponentsWithStats(acc, connectivity=8)
holds = []
for i in range(1, num):
    x, y, w, h, area = stats[i]
    if area < 200:  # 노이즈 컷
        continue
    cx, cy = centroids[i]
    r = int(0.5 * (w + h) / 2)
    holds.append((int(cx), int(cy), r))

print(f"검출된 홀드 개수: {len(holds)}")


# ------------------------------
# 6) 시각화 결과 저장
# ------------------------------
vis = frame0.copy()
for (cx, cy, r) in holds:
    cv.circle(vis, (cx, cy), r, (0, 255, 0), 2)

cv.imwrite("route_overlay.png", vis)
cv.imwrite("accumulated_mask.png", acc)

print("route_overlay.png / accumulated_mask.png 저장 완료")
