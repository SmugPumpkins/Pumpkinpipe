from pumpkinpipe.utils.model_loader import  get_model_path
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
import cv2, math
import numpy as np

class PoseDetector:
    def __init__(self, max_poses=1):
        with get_model_path("pose_landmarker.task") as model_path:
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path = model_path
                ),
                num_poses=max_poses,
                running_mode=vision.RunningMode.VIDEO
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.timestamp_ms = 0
            self.frame_rate = 30

    def find_poses(self, image: np.ndarray, flip: bool=False) -> list[Pose]:
        pass

class Pose:
    def __init__(self):
        pass