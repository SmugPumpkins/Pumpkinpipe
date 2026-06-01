"""
Face Detection Module

Author: Nathan Forsyth
"""
import cv2
import math
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from pumpkinpipe.utils.drawing import BoundingBox, ConnectionStyle, LandmarkStyle
from pumpkinpipe.utils.model_loader import get_model_path
from pumpkinpipe.utils.text import HAlign, VAlign, stack_text


class Face:
    """
    Class for displaying and retrieving data for faces recognized by the FaceDetector module.

    :ivar landmarks: List of landmarks using 3D pixel coordinates.
    :ivar normalized_landmarks: List of the original normalized landmark tuples provided by MediaPipe.
    :ivar box: Bounding box of the face.
    :ivar center: 2D pixel coordinates of the bounding box center.
    :ivar flip: When True, left/right feature shortcuts are swapped for mirrored images.
    :ivar left_eye_open: Whether the left eye is detected as open.
    :ivar right_eye_open: Whether the right eye is detected as open.
    :ivar mouth_open: Whether the mouth is detected as open.
    :ivar left_iris_center: 2D pixel center of the left iris.
    :ivar right_iris_center: 2D pixel center of the right iris.
    :ivar iris_tracking: Dictionary of iris positions inside each eye as normalized (x, y) tuples.
    :ivar image: Image used by the FaceDetector. Used as the default image to display the drawn face.
    :ivar connection_style: Style settings for face connection lines.
    :ivar landmark_style: Style settings for face landmarks.
    """

    _DEFAULT_CONNECTIONS = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    _LEFT_EYE_OPEN_IDS = (362, 385, 387, 263, 373, 380)
    _RIGHT_EYE_OPEN_IDS = (33, 160, 158, 133, 153, 144)
    _MOUTH_OPEN_IDS = (13, 14, 61, 291)
    _EYE_OPEN_THRESHOLD = 0.2
    _MOUTH_OPEN_THRESHOLD = 0.15

    def __init__(
        self,
        landmarks: list[tuple[int, int, int]],
        normalized_landmarks: list[tuple[float, float, float]],
        box: BoundingBox,
        image: np.ndarray,
        flip: bool = True,
    ):
        """
        Initialize the face.

        :param landmarks: The (x, y, z) pixel coordinates of landmarks.
        :param normalized_landmarks: The normalized landmark tuples provided by MediaPipe.
        :param box: The bounding box of the face.
        :param image: The image that was used to detect the face.
        :param flip: When True, swap left/right feature shortcuts for mirrored images.
        """
        self.landmarks: list[tuple[int, int, int]] = landmarks
        self.normalized_landmarks: list[tuple[float, float, float]] = normalized_landmarks
        self.original_landmarks: list[tuple[float, float, float]] = normalized_landmarks
        self.flip: bool = flip
        self.box: BoundingBox = box
        self.center: tuple[int, int] = self.box.center
        self.image: np.ndarray = image

        self.left_eye, self.right_eye = self._side_feature_landmarks(
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
            FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
        )
        self.left_eyebrow, self.right_eyebrow = self._side_feature_landmarks(
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
            FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW,
        )
        self.left_iris, self.right_iris = self._side_feature_landmarks(
            FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
        )
        left_eye_open_ids, right_eye_open_ids = self._side_ids(
            Face._LEFT_EYE_OPEN_IDS,
            Face._RIGHT_EYE_OPEN_IDS,
        )
        self.left_eye_open: bool = self._eye_is_open(left_eye_open_ids)
        self.right_eye_open: bool = self._eye_is_open(right_eye_open_ids)
        self.mouth_open: bool = self._mouth_is_open()
        self.left_iris_center: tuple[int, int] | None = self._center_of(self.left_iris)
        self.right_iris_center: tuple[int, int] | None = self._center_of(self.right_iris)
        self.iris_tracking: dict[str, tuple[float, float] | None] = {
            "left": self._iris_position(self.left_iris_center, self.left_eye),
            "right": self._iris_position(self.right_iris_center, self.right_eye),
        }

        self.connection_style: ConnectionStyle = ConnectionStyle()
        self.landmark_style: LandmarkStyle = LandmarkStyle(fill=(0, 255, 255), radius=1)

    def _side_ids(self, left_ids, right_ids):
        if self.flip:
            return right_ids, left_ids
        return left_ids, right_ids

    def _side_feature_landmarks(self, left_connections, right_connections):
        if self.flip:
            left_connections, right_connections = right_connections, left_connections
        return (
            self._landmarks_for_connections(left_connections),
            self._landmarks_for_connections(right_connections),
        )

    def _landmarks_for_connections(self, connections) -> list[tuple[int, int, int]]:
        landmark_ids = sorted({connection.start for connection in connections} | {connection.end for connection in connections})
        return [self.landmarks[index] for index in landmark_ids if index < len(self.landmarks)]

    def _eye_is_open(self, ids: tuple[int, int, int, int, int, int]) -> bool:
        if max(ids) >= len(self.landmarks):
            return False
        p1, p2, p3, p4, p5, p6 = [self.landmarks[index] for index in ids]
        vertical_distance = math.dist(p2[:2], p6[:2]) + math.dist(p3[:2], p5[:2])
        horizontal_distance = 2 * math.dist(p1[:2], p4[:2])
        if horizontal_distance == 0:
            return False
        return vertical_distance / horizontal_distance > Face._EYE_OPEN_THRESHOLD

    def _mouth_is_open(self) -> bool:
        if max(Face._MOUTH_OPEN_IDS) >= len(self.landmarks):
            return False
        upper_lip = self.landmarks[13]
        lower_lip = self.landmarks[14]
        left_corner = self.landmarks[61]
        right_corner = self.landmarks[291]
        mouth_width = math.dist(left_corner[:2], right_corner[:2])
        if mouth_width == 0:
            return False
        return math.dist(upper_lip[:2], lower_lip[:2]) / mouth_width > Face._MOUTH_OPEN_THRESHOLD

    def _center_of(self, landmarks: list[tuple[int, int, int]]) -> tuple[int, int] | None:
        if not landmarks:
            return None
        x = sum(point[0] for point in landmarks) // len(landmarks)
        y = sum(point[1] for point in landmarks) // len(landmarks)
        return x, y

    def _iris_position(
        self,
        iris_center: tuple[int, int] | None,
        eye_landmarks: list[tuple[int, int, int]],
    ) -> tuple[float, float] | None:
        if iris_center is None or not eye_landmarks:
            return None
        min_x = min(point[0] for point in eye_landmarks)
        max_x = max(point[0] for point in eye_landmarks)
        min_y = min(point[1] for point in eye_landmarks)
        max_y = max(point[1] for point in eye_landmarks)
        width = max_x - min_x
        height = max_y - min_y
        if width == 0 or height == 0:
            return None
        x, y = iris_center
        return (x - min_x) / width, (y - min_y) / height

    def draw(
        self,
        image: np.ndarray | None = None,
        connections=None,
        landmarks: bool = True,
    ):
        """
        Draws the face landmarks and connections on the specified image.

        :param image: The target image for drawing. If None, it will draw on the face's image.
        :param connections: MediaPipe face connection set to draw. Defaults to face contours.
        :param landmarks: When True, draw landmark points.
        """
        if image is None:
            image = self.image
        if connections is None:
            connections = Face._DEFAULT_CONNECTIONS

        for connection in connections:
            start_x, start_y, _ = self.landmarks[connection.start]
            end_x, end_y, _ = self.landmarks[connection.end]
            cv2.line(
                image,
                (start_x, start_y),
                (end_x, end_y),
                self.connection_style.stroke,
                self.connection_style.thickness,
            )

        if landmarks:
            for x, y, _ in self.landmarks:
                cv2.circle(image, (x, y), self.landmark_style.radius, self.landmark_style.fill, -1)

    def debug(
        self,
        image: np.ndarray | None = None,
        contours: bool = True,
        bounding_box: bool = True,
        center: bool = True,
        landmark_count: bool = True,
        face_status: bool = True,
        iris_tracking: bool = True,
    ):
        """
        Draws requested debug information. Defaults to all debug information.

        :param image: The image to draw on. If None, the face will draw on its own image.
        :param contours: When True, draw face contours and landmark points.
        :param bounding_box: When True, draw the outer bounding box of the face.
        :param center: When True, draw and label the bounding box center.
        :param landmark_count: When True, display the number of landmarks found.
        :param face_status: When True, display eye and mouth open/closed status.
        :param iris_tracking: When True, draw iris centers and display iris tracking values.
        """
        if image is None:
            image = self.image

        debug_font = cv2.FONT_HERSHEY_PLAIN
        debug_text_size = 1
        debug_thickness = 1

        if contours:
            self.draw(image)

        if bounding_box:
            self.box.draw_corners(image, length=20, thickness=5, stroke=(0, 0, 0))
            self.box.draw_corners(image, length=19, thickness=4, stroke=(255, 255, 255))

        if landmark_count:
            stack_text(
                image,
                [f"Face landmarks: {len(self.landmarks)}"],
                self.box.origin,
                debug_font,
                debug_text_size,
                debug_thickness,
                (255, 255, 255),
                HAlign.RIGHT,
                VAlign.TOP,
            )

        debug_lines = []
        if face_status:
            debug_lines.extend(
                [
                    f"Left eye: {'Open' if self.left_eye_open else 'Closed'}",
                    f"Right eye: {'Open' if self.right_eye_open else 'Closed'}",
                    f"Mouth: {'Open' if self.mouth_open else 'Closed'}",
                ]
            )
        if iris_tracking:
            for label, tracking in self.iris_tracking.items():
                if tracking is None:
                    debug_lines.append(f"{label.title()} iris: n/a")
                else:
                    x, y = tracking
                    debug_lines.append(f"{label.title()} iris: ({x:.2f}, {y:.2f})")

            for iris_center in (self.left_iris_center, self.right_iris_center):
                if iris_center is not None:
                    cv2.circle(image, iris_center, 4, (255, 0, 255), -1)
                    cv2.circle(image, iris_center, 4, (0, 0, 0), 1)

        if debug_lines:
            stack_text(
                image,
                debug_lines,
                self.box.opposite,
                debug_font,
                debug_text_size,
                debug_thickness,
                (255, 255, 255),
                HAlign.RIGHT,
                VAlign.TOP,
            )

        if center:
            cv2.circle(image, self.center, 7, (0, 255, 0), -1)
            cv2.circle(image, self.center, 7, (0, 0, 0), 2)
            stack_text(
                image,
                [f"Center: ({self.center[0]}, {self.center[1]})"],
                self.box.origin,
                debug_font,
                debug_text_size * 1.5,
                debug_thickness * 2,
                (0, 255, 0),
                HAlign.LEFT,
                VAlign.BOTTOM,
                0,
            )

    def set_connection_style(
        self,
        stroke: tuple[int, int, int] | list[int] | None = None,
        thickness: float | None = None,
    ):
        """
        Modifies the style of the face connections when drawn.

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
        Modifies the style of the face landmarks when drawn.

        :param fill: The BGR color of the landmarks.
        :param stroke: The BGR color of the landmark outlines. Stored for API consistency.
        :param radius: Radius of the landmarks.
        :param thickness: Thickness of the landmark outline. Stored for API consistency.
        """
        if fill is not None:
            self.landmark_style.fill = fill
        if stroke is not None:
            self.landmark_style.stroke = stroke
        if radius is not None:
            self.landmark_style.radius = int(radius)
        if thickness is not None:
            self.landmark_style.thickness = int(thickness)


class FaceDetector:
    """
    Class for setting up MediaPipe and finding face data.

    :ivar timestamp_ms: Variable for tracking time between frames.
    :ivar frame_rate: Desired frame rate. Set to 30.
    :ivar landmarker: Pipeline to detect face landmarks.
    """

    def __init__(self, number_of_faces: int = 1):
        """
        Initialize the face detector.

        :param number_of_faces: The maximum number of faces for the detector to try and process.
        """
        with get_model_path("face_landmarker.task") as model_path:
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=number_of_faces,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.frame_rate = 30

    def find_faces(self, image: np.ndarray, flip: bool = True) -> list[Face]:
        """
        Detect faces and add them to a list.

        :param image: The image to detect faces in.
        :param flip: When True, swap left/right feature shortcuts for mirrored images.
        :return: A list of detected faces.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        self.timestamp_ms += int(1000 / self.frame_rate)

        if not result.face_landmarks:
            return []

        faces: list[Face] = []
        for face_lms in result.face_landmarks:
            pixel_landmarks: list[tuple[int, int, int]] = []
            normalized_landmarks: list[tuple[float, float, float]] = []
            x_list: list[int] = []
            y_list: list[int] = []

            for lm in face_lms:
                x = int(lm.x * width)
                y = int(lm.y * height)
                z = int(lm.z * width)
                normalized_landmarks.append((lm.x, lm.y, lm.z))
                pixel_landmarks.append((x, y, z))
                x_list.append(x)
                y_list.append(y)

            bounding_box = BoundingBox(
                (min(x_list) - 10, min(y_list) - 10),
                (max(x_list) + 10, max(y_list) + 10),
            )
            faces.append(Face(pixel_landmarks, normalized_landmarks, bounding_box, image, flip))

        return faces


def main():
    """
    Test script for the Face Detection module.
    """
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()

    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)

        faces = face_detector.find_faces(img)
        for face in faces:
            face.debug()

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
