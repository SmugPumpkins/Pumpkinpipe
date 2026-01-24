# MediaPipe Tasks – Image Segmenter (Selfie Segmentation, FIXED)

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_PATH = "../../pumpkinpipe/models/selfie_segmenter.tflite"

options = vision.ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    output_confidence_masks=True,
    output_category_mask=False
)

segmenter = vision.ImageSegmenter.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=rgb
    )

    result = segmenter.segment_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    if result.confidence_masks:
        # Selfie model has ONE mask
        mask = result.confidence_masks[0].numpy_view()

        # Convert [0.0–1.0] → [0–255]
        mask = (mask * 255).astype(np.uint8)

        # Resize defensively
        mask = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Threshold to get binary person mask
        _, person_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        # Visualize
        overlay = frame.copy()
        overlay[person_mask == 255] = (0, 255, 0)

        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        cv2.imshow("Person Mask", person_mask)

    cv2.imshow("Selfie Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
segmenter.close()
