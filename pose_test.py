# MediaPipe Tasks – Pose Landmarker (Python, VIDEO mode)
# Model path: src/pumpkinpipe/models/pose_landmarker_full.task

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = "src/pumpkinpipe/models/pose_landmarker_full.task"

# --- Configure Pose Landmarker ---
options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    output_segmentation_masks=False,
)

landmarker = vision.PoseLandmarker.create_from_options(options)

# --- Open camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap frame for MediaPipe Tasks
    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    # Run pose detection
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33  # ~30 FPS

    # Draw pose landmarks
    if result.pose_landmarks:
        h, w, _ = frame.shape
        for pose in result.pose_landmarks:
            for lm in pose:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow("Pose Landmarks (MediaPipe Tasks)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
