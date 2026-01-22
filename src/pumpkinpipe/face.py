import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from pumpkinpipe.utils.model_loader import get_model_path

class Face:
    def __init__(self, landmarks, original_landmarks):
        self.landmarks = landmarks
        self.original_landmarks = original_landmarks

    def draw(self):
        pass

    def debug(self):
        pass

class FaceDetector:
    def __init__(self, number_of_faces=1):
        with get_model_path("hand_landmarker.task") as model_path:
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=model_path
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=number_of_faces,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.frame_rate = 30

    def find_faces(self, image, flip=False):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=image_rgb
        )
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        self.timestamp_ms += int(1000/self.frame_rate)

        if not result.face_landmarks:
            return []
        faces : list[Face] = []
        for face_lms in result.face_landmarks:
            pixel_landmarks = []
            for landmark in face_lms:
                px_lm = (int(landmark.x * width), int(landmark.y * height), int(landmark.z * width))
                pixel_landmarks.append(px_lm)
            pixel_landmarks = tuple(pixel_landmarks)
            face = Face(pixel_landmarks, face_lms)
            faces.append(face)
        return faces