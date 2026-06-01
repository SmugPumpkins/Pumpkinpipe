"""
Pose Detection Module

Author: Nathan Forsyth
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarksConnections

from pumpkinpipe.utils.drawing import BoundingBox, ConnectionStyle, LandmarkStyle
from pumpkinpipe.utils.model_loader import get_model_path
from pumpkinpipe.utils.text import HAlign, VAlign, stack_text


class Pose:
    """
    Class for displaying and retrieving data for poses recognized by the PoseDetector module.

    :ivar landmarks: List of landmarks using 3D pixel coordinates.
    :ivar normalized_landmarks: List of the original normalized landmark tuples provided by MediaPipe.
    :ivar world_landmarks: List of world landmark tuples in meters when MediaPipe provides them.
    :ivar box: Bounding box of the pose.
    :ivar center: 2D pixel coordinates of the bounding box center.
    :ivar flip: When True, left/right landmark shortcuts are swapped for mirrored images.
    :ivar image: Image used by the PoseDetector. Used as the default image to display the drawn pose.
    :ivar connection_style: Style settings for the pose connection lines.
    :ivar landmark_style: Style settings for the pose landmarks.
    """

    _CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS

    _NOSE_ID = 0
    _LEFT_SHOULDER_ID = 11
    _RIGHT_SHOULDER_ID = 12
    _LEFT_ELBOW_ID = 13
    _RIGHT_ELBOW_ID = 14
    _LEFT_WRIST_ID = 15
    _RIGHT_WRIST_ID = 16
    _LEFT_HIP_ID = 23
    _RIGHT_HIP_ID = 24
    _LEFT_KNEE_ID = 25
    _RIGHT_KNEE_ID = 26
    _LEFT_ANKLE_ID = 27
    _RIGHT_ANKLE_ID = 28

    def __init__(
        self,
        landmarks: list[tuple[int, int, int]],
        normalized_landmarks: list[tuple[float, float, float]],
        world_landmarks: list[tuple[float, float, float]],
        box: BoundingBox,
        image: np.ndarray,
        flip: bool = True,
    ):
        """
        Initialize the pose.

        :param landmarks: The (x, y, z) pixel coordinates of landmarks.
        :param normalized_landmarks: The normalized landmark tuples provided by MediaPipe.
        :param world_landmarks: The world landmark tuples provided by MediaPipe.
        :param box: The bounding box of the pose.
        :param image: The image that was used to detect the pose.
        :param flip: When True, swap left/right shortcuts for mirrored images.
        """
        self.landmarks: list[tuple[int, int, int]] = landmarks
        self.normalized_landmarks: list[tuple[float, float, float]] = normalized_landmarks
        self.world_landmarks: list[tuple[float, float, float]] = world_landmarks
        self.flip: bool = flip

        self.nose: tuple[int, int, int] = self.landmarks[Pose._NOSE_ID]
        left_shoulder_id, right_shoulder_id = self._side_ids(Pose._LEFT_SHOULDER_ID, Pose._RIGHT_SHOULDER_ID)
        left_elbow_id, right_elbow_id = self._side_ids(Pose._LEFT_ELBOW_ID, Pose._RIGHT_ELBOW_ID)
        left_wrist_id, right_wrist_id = self._side_ids(Pose._LEFT_WRIST_ID, Pose._RIGHT_WRIST_ID)
        left_hip_id, right_hip_id = self._side_ids(Pose._LEFT_HIP_ID, Pose._RIGHT_HIP_ID)
        left_knee_id, right_knee_id = self._side_ids(Pose._LEFT_KNEE_ID, Pose._RIGHT_KNEE_ID)
        left_ankle_id, right_ankle_id = self._side_ids(Pose._LEFT_ANKLE_ID, Pose._RIGHT_ANKLE_ID)

        self.left_shoulder: tuple[int, int, int] = self.landmarks[left_shoulder_id]
        self.right_shoulder: tuple[int, int, int] = self.landmarks[right_shoulder_id]
        self.left_elbow: tuple[int, int, int] = self.landmarks[left_elbow_id]
        self.right_elbow: tuple[int, int, int] = self.landmarks[right_elbow_id]
        self.left_wrist: tuple[int, int, int] = self.landmarks[left_wrist_id]
        self.right_wrist: tuple[int, int, int] = self.landmarks[right_wrist_id]
        self.left_hip: tuple[int, int, int] = self.landmarks[left_hip_id]
        self.right_hip: tuple[int, int, int] = self.landmarks[right_hip_id]
        self.left_knee: tuple[int, int, int] = self.landmarks[left_knee_id]
        self.right_knee: tuple[int, int, int] = self.landmarks[right_knee_id]
        self.left_ankle: tuple[int, int, int] = self.landmarks[left_ankle_id]
        self.right_ankle: tuple[int, int, int] = self.landmarks[right_ankle_id]

        self.box: BoundingBox = box
        self.center: tuple[int, int] = self.box.center
        self.image: np.ndarray = image

        self.connection_style: ConnectionStyle = ConnectionStyle()
        self.landmark_style: LandmarkStyle = LandmarkStyle(fill=(0, 255, 0))

    def _side_ids(self, left_id: int, right_id: int) -> tuple[int, int]:
        if self.flip:
            return right_id, left_id
        return left_id, right_id

    def draw(self, image: np.ndarray | None = None):
        """
        Draws the pose skeleton on the specified image.

        :param image: The target image for the drawing. If None, it will draw on the pose's image.
        """
        if image is None:
            image = self.image

        for connection in Pose._CONNECTIONS:
            start_x, start_y, _ = self.landmarks[connection.start]
            end_x, end_y, _ = self.landmarks[connection.end]
            cv2.line(
                image,
                (start_x, start_y),
                (end_x, end_y),
                self.connection_style.stroke,
                self.connection_style.thickness,
            )

        for x, y, _ in self.landmarks:
            cv2.circle(image, (x, y), self.landmark_style.radius, self.landmark_style.fill, -1)
            cv2.circle(
                image,
                (x, y),
                self.landmark_style.radius,
                self.landmark_style.stroke,
                self.landmark_style.thickness,
            )

    def debug(
        self,
        image: np.ndarray | None = None,
        skeleton: bool = True,
        bounding_box: bool = True,
        center: bool = True,
        landmark_count: bool = True,
        key_points: bool = True,
    ):
        """
        Draws the requested debug information. Defaults to all debug information.

        :param image: The image to draw on. If None, the pose will draw on its own image.
        :param skeleton: When True, draw the pose landmarks and connections.
        :param bounding_box: When True, draw the outer bounding box of the pose.
        :param center: When True, draw and label the bounding box center.
        :param landmark_count: When True, display the number of landmarks found.
        :param key_points: When True, label common pose landmarks.
        """
        if image is None:
            image = self.image

        debug_font = cv2.FONT_HERSHEY_PLAIN
        debug_text_size = 1
        debug_thickness = 1

        if skeleton:
            self.draw(image)

        if bounding_box:
            self.box.draw_corners(image, length=25, thickness=5, stroke=(0, 0, 0))
            self.box.draw_corners(image, length=24, thickness=4, stroke=(255, 255, 255))

        if landmark_count:
            stack_text(
                image,
                [f"Pose landmarks: {len(self.landmarks)}"],
                self.box.origin,
                debug_font,
                debug_text_size,
                debug_thickness,
                (255, 255, 255),
                HAlign.LEFT,
                VAlign.BOTTOM,
            )

        if key_points:
            labels = [
                ("nose", self.nose),
                ("L shoulder", self.left_shoulder),
                ("R shoulder", self.right_shoulder),
                ("L wrist", self.left_wrist),
                ("R wrist", self.right_wrist),
                ("L hip", self.left_hip),
                ("R hip", self.right_hip),
                ("L ankle", self.left_ankle),
                ("R ankle", self.right_ankle),
            ]
            for label, point in labels:
                x, y, _ = point
                stack_text(
                    image,
                    [label],
                    (x, y),
                    debug_font,
                    debug_text_size,
                    debug_thickness,
                    (0, 255, 0),
                    HAlign.CENTER,
                    VAlign.BOTTOM,
                    0,
                )

        if center:
            cv2.circle(image, self.center, 7, (0, 255, 0), -1)
            cv2.circle(image, self.center, 7, (0, 0, 0), 2)
            stack_text(
                image,
                [f"Center: ({self.center[0]}, {self.center[1]})"],
                self.center,
                debug_font,
                debug_text_size * 1.5,
                debug_thickness * 2,
                (0, 255, 0),
                HAlign.CENTER,
                VAlign.BOTTOM,
                0,
            )

    def set_connection_style(
        self,
        stroke: tuple[int, int, int] | list[int] | None = None,
        thickness: float | None = None,
    ):
        """
        Modifies the style of the pose connections when drawn.

        :param stroke: The BGR color of the connections.
        :param thickness: The thickness of the connector lines in pixels.
        """
        if stroke is not None:
            self.connection_style.stroke = stroke
        if thickness is not None:
            self.connection_style.thickness = int(thickness)

    def set_landmarks_style(
        self,
        fill: tuple[int, int, int] | list[int] | None = None,
        stroke: tuple[int, int, int] | list[int] | None = None,
        radius: float | None = None,
        thickness: float | None = None,
    ):
        """
        Modifies the style of the pose landmarks when drawn.

        :param fill: The BGR color of the landmarks.
        :param stroke: The BGR color of the landmark outlines.
        :param radius: Radius of the landmarks.
        :param thickness: Thickness of the landmark outline.
        """
        if fill is not None:
            self.landmark_style.fill = fill
        if stroke is not None:
            self.landmark_style.stroke = stroke
        if radius is not None:
            self.landmark_style.radius = int(radius)
        if thickness is not None:
            self.landmark_style.thickness = int(thickness)


class PoseDetector:
    """
    Class for setting up MediaPipe and finding pose data.

    :ivar timestamp_ms: Variable for tracking time between frames.
    :ivar frame_rate: Desired frame rate. Set to 30.
    :ivar landmarker: Pipeline to detect pose landmarks.
    """

    def __init__(self, max_poses: int = 1):
        """
        Initialize the pose detector.

        :param max_poses: The maximum number of poses for the detector to try and process.
        """
        with get_model_path("pose_landmarker_full.task") as model_path:
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                num_poses=max_poses,
                running_mode=vision.RunningMode.VIDEO,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.frame_rate = 30

    def find_poses(self, image: np.ndarray, flip: bool = True) -> list[Pose]:
        """
        Detect poses and add them to a list.

        :param image: The image to detect poses in.
        :param flip: When True, swap left/right shortcuts for mirrored images.
        :return: A list of detected poses.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        self.timestamp_ms += int(1000 / self.frame_rate)

        if not result.pose_landmarks:
            return []

        poses: list[Pose] = []
        world_landmark_sets = result.pose_world_landmarks or [[] for _ in result.pose_landmarks]

        for landmarks, world_landmarks in zip(result.pose_landmarks, world_landmark_sets):
            pixel_landmarks: list[tuple[int, int, int]] = []
            normalized_landmarks: list[tuple[float, float, float]] = []
            pose_world_landmarks: list[tuple[float, float, float]] = []
            x_list: list[int] = []
            y_list: list[int] = []

            for lm in landmarks:
                x = int(lm.x * width)
                y = int(lm.y * height)
                z = int(lm.z * width)
                normalized_landmarks.append((lm.x, lm.y, lm.z))
                pixel_landmarks.append((x, y, z))
                x_list.append(x)
                y_list.append(y)

            for lm in world_landmarks:
                pose_world_landmarks.append((lm.x, lm.y, lm.z))

            bounding_box = BoundingBox(
                (min(x_list) - 10, min(y_list) - 10),
                (max(x_list) + 10, max(y_list) + 10),
            )
            poses.append(Pose(pixel_landmarks, normalized_landmarks, pose_world_landmarks, bounding_box, image, flip))

        return poses


def main():
    """
    Test script for the Pose Detection module.
    """
    cap = cv2.VideoCapture(0)
    pose_detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        poses = pose_detector.find_poses(img)
        for pose in poses:
            pose.debug()

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
