"""
Hand Detection Module
Author: Nathan Forsyth
"""
from dataclasses import dataclass
from typing import Tuple

import cv2, math
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from pumpkinpipe.utils.model_loader import get_model_path
from pumpkinpipe.utils.drawing import BoundingBox, LandmarkStyle, ConnectionStyle
from pumpkinpipe.utils.text import stack_text, HAlign, VAlign
from mediapipe.tasks.python.vision import HandLandmarksConnections



def angle_3d(p1, p2):
    """
    Returns the normalized 3D vector of 2 points.
    :param p1: The origin point
    :param p2: The offset point
    :return: Normalized 3D point representing the angle between 2 3D points
    """
    # vector 1 → 2
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    z = p2[2] - p1[2]
    # vector magnitude (length)
    magnitude = math.sqrt(x * x + y * y + z * z)
    # avoid division by zero
    if magnitude == 0:
        return 0, 0, 0
    # normalized vector (unit length)
    return x / magnitude, y / magnitude, z / magnitude

@dataclass
class Line:
    start: Tuple[int, int]
    end: Tuple[int, int]
    z: float
    color: Tuple[int, int, int]

@dataclass
class Point:
    pos: Tuple[int, int]
    z: int
    color: Tuple[int, int, int]


class Hand:

    DEFAULT_CONNECTION_STYLE = ConnectionStyle()
    DEFAULT_LANDMARK_STYLE = LandmarkStyle()

    REGION_COLORS = (
        (245, 135, 66),
        (245, 66, 167),
        (105, 66, 245),
        (66, 152, 245),
        (66, 245, 176),
        (127, 127, 127)
    )

    CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS
    PALM_CONNECTIONS = HandLandmarksConnections.HAND_PALM_CONNECTIONS
    THUMB_CONNECTIONS = HandLandmarksConnections.HAND_THUMB_CONNECTIONS
    INDEX_CONNECTIONS = HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS
    MIDDLE_CONNECTIONS = HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS
    RING_CONNECTIONS = HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS
    PINKY_CONNECTIONS = HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS

    THUMB_LANDMARKS = (2, 3, 4)
    INDEX_LANDMARKS = (6, 7, 8)
    MIDDLE_LANDMARKS = (10, 11, 12)
    RING_LANDMARKS = (14, 15, 16)
    PINKY_LANDMARKS = (18, 19, 20)
    PALM_LANDMARKS = (0, 1, 5, 9, 13, 17)

    THUMB_TIP_ID = 4
    INDEX_TIP_ID = 8
    MIDDLE_TIP_ID = 12
    RING_TIP_ID = 16
    PINKY_TIP_ID = 20
    WRIST_ID = 0


    def __init__(self, landmarks, original_landmarks, side, box : BoundingBox, image):
        """
        Stores hand data for later use.
        :param landmarks: The [x,y,z] pixel coordinates of landmarks.
        :param original_landmarks: The actual landmarks of the hand as provided by mediapipe.
        :param side: The side of the hand ("Left" or "Right").
        :param box: The bounding box of the hand.
        """
        self.landmarks = landmarks
        self.original_landmarks = original_landmarks
        self.side = side
        self.thumb = self.landmarks[Hand.THUMB_TIP_ID]
        self.index = self.landmarks[Hand.INDEX_TIP_ID]
        self.middle = self.landmarks[Hand.MIDDLE_TIP_ID]
        self.ring = self.landmarks[Hand.RING_TIP_ID]
        self.pinky = self.landmarks[Hand.PINKY_TIP_ID]
        self.wrist = self.landmarks[Hand.WRIST_ID]
        self.connection_style = ConnectionStyle()
        self.landmark_style = LandmarkStyle()

        self.flags = self.finger_flags()
        self.box = box
        self.center = self.box.center

        self.image = image


    def landmark_distance(self, landmark_index_1, landmark_index_2, image=None, draw=False):
        """
        Finds the distance in pixels between 2 specified landmarks.
        :param landmark_index_1: The landmark index for the first point
        :param landmark_index_2: The landmark index for the second point
        :param image: If not None, draws a line between the 2 points on the specified image
        :param draw: If True, currently does nothing. Future implementations will draw the line.
        :return: The distance between 2 points
        """
        landmark_1 = self.landmarks[landmark_index_1]
        landmark_2 = self.landmarks[landmark_index_2]
        distance : float = math.dist(landmark_1, landmark_2)
        if draw:
            if image is not None:
                pass
        return distance

    def finger_flags(self):
        """
        Finds which fingers are up and returns them as a binary list in the order of thumb, index, middle, ring, pinky.
        :return: A list of binary values representing whether a finger is up or down
        """
        # Initialize empty list for finger flags
        fingers = []

        # Distance for thumb to be considered open
        distance_threshold = 0.3

        # Vector math to determine whether landmarks 1→2 are closely aligned with 2→3
        angle_a = angle_3d(self.landmarks[1], self.landmarks[2])
        angle_b = angle_3d(self.landmarks[2], self.landmarks[3])
        thumb_angle_distance = math.dist(angle_a, angle_b)
        # Append thumb value to fingers
        if thumb_angle_distance < distance_threshold:
            fingers.append(1)
        else:
            fingers.append(0)

        # Landmark indices for index_tip, middle_tip, ring_tip, and pinky_tip
        finger_indices = [Hand.INDEX_TIP_ID, Hand.MIDDLE_TIP_ID, Hand.RING_TIP_ID, Hand.PINKY_TIP_ID]

        # Compare the distance between the tip and the wrist to the distance between the knuckle and the wrist
        for index in finger_indices:
            tip_distance = math.dist(self.landmarks[index], self.landmarks[Hand.WRIST_ID])
            knuckle_distance = math.dist(self.landmarks[index - 2], self.landmarks[Hand.WRIST_ID])
            # Append each finger value to fingers
            if tip_distance > knuckle_distance:
                fingers.append(1)
            else:
                fingers.append(0)

        # Return list of flags
        return fingers

    def fingers_up(self):
        """
        Provides a list of the fingers that are up in English.
        """
        # Initialize empty list for finger flags
        fingers = []
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        for flag, name in zip(self.flags, finger_names):
            # Append fingers
            if flag > 0:
                fingers.append(name)

        # Return list of fingers that are up
        return fingers

    def fingers_down(self):
        """
        Provides a list of the fingers that are down in English.
        """
        # Initialize empty list for finger flags
        fingers = []
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        for flag, name in zip(self.flags, finger_names):
            # Append fingers
            if flag < 1:
                fingers.append(name)
        # Return list of fingers that are up
        return fingers

    def draw(self, image=None):
        """
        Draws the hand skeleton on the specified image.
        Currently, this library uses a custom way to draw while
        drawing_utils.py remains unimplemented in the
        official mediapipe release.
        :param image: The target image for the drawing. If none, it will draw on the hands self.image
        """
        if image is None:
            image = self.image

        for connection in Hand.CONNECTIONS:
            start_x, start_y, _ = self.landmarks[connection.start]
            end_x, end_y, _ = self.landmarks[connection.end]
            cv2.line(
                image,
                (start_x, start_y),
                (end_x, end_y),
                self.connection_style.stroke,
                self.connection_style.thickness
            )
        for landmark in self.landmarks:
            x, y, _ = landmark
            # Filled circle
            cv2.circle(
                image,
                (x,y),
                self.landmark_style.radius,
                self.landmark_style.fill,
                -1
            )
            # Outline
            cv2.circle(
                image,
                (x, y),
                self.landmark_style.radius,
                self.landmark_style.stroke,
                self.landmark_style.thickness
            )

    def debug(self, image=None, skeleton=True, bounding_box=True, center=True, side=True, fingers=True, flags=True, tip_points=True):
        """
        Draws the requested debug information. Defaults to all debug information.
        :param image: The image to draw the debug information on. If image is none then the hand will draw on its self.image
        :param skeleton: When True, draw the landmarks and connections of the hand. Hand regions are separated by color.
        :param bounding_box: When True, draw the outer bounding box of the hand.
        :param center: When True, draw a green circle at the center of the bounding box. Also display the value for hand.center.
        :param side: When True, write the value for hand.side underneath the hand.
        :param flags: When True, display the value for hand.flags (binary list) representing which fingers are up or down.
        :param fingers: When True, display the value returned by hand.fingers_up() (a list of strings for each finger that is registered as being up).
        :param tip_points: When True, display hand.thumb, hand.index, hand.middle, hand.ring, hand.pinky, and hand.wrist values near their corresponding fingertips.
        """

        if image is None:
            image = self.image

        # Set default values
        height, width, _ = image.shape
        debug_text_size = 1
        debug_font = cv2.FONT_HERSHEY_PLAIN
        debug_thickness = 1

        # Display the hand connections and landmarks with the different regions as different colors.
        if skeleton:
            connection_lines = []
            for connections, color in zip(
                [
                    Hand.THUMB_CONNECTIONS,
                    Hand.INDEX_CONNECTIONS,
                    Hand.MIDDLE_CONNECTIONS,
                    Hand.RING_CONNECTIONS,
                    Hand.PINKY_CONNECTIONS,
                    Hand.PALM_CONNECTIONS
                ],
                Hand.REGION_COLORS
            ):
                for connection in connections:
                    start_x, start_y, start_z = self.landmarks[connection.start]
                    end_x, end_y, end_z = self.landmarks[connection.end]
                    z_average = -(start_z + end_z) / 2
                    new_line : Line = Line((start_x, start_y), (end_x, end_y), z_average, color)
                    connection_lines.append(new_line)
            points = []
            for landmarks, color in zip(
                [
                    Hand.THUMB_LANDMARKS,
                    Hand.INDEX_LANDMARKS,
                    Hand.MIDDLE_LANDMARKS,
                    Hand.RING_LANDMARKS,
                    Hand.PINKY_LANDMARKS,
                    Hand.PALM_LANDMARKS
                ],
                Hand.REGION_COLORS
            ):
                for landmark in landmarks:
                    x, y, z = self.landmarks[landmark]
                    points.append(Point((x, y,), -z, color))
            connection_lines.sort(key=lambda obj: obj.z)
            for connection_line in connection_lines:
                cv2.line(
                    image,
                    connection_line.start,
                    connection_line.end,
                    connection_line.color,
                    Hand.DEFAULT_CONNECTION_STYLE.thickness
                )
            points.sort(key=lambda obj: obj.z)
            for point in points:
                cv2.circle(
                    image,
                    point.pos,
                    Hand.DEFAULT_LANDMARK_STYLE.radius,
                    point.color,
                    -1
                )
                cv2.circle(
                    image,
                    point.pos,
                    Hand.DEFAULT_LANDMARK_STYLE.radius,
                    Hand.DEFAULT_LANDMARK_STYLE.stroke,
                    Hand.DEFAULT_LANDMARK_STYLE.thickness
                )

        # Display the bounding box
        if bounding_box:
            self.box.draw_corners(image, length=20, thickness=5, stroke=(0,0,0))
            self.box.draw_corners(image, length=19, thickness=4, stroke=(255,255,255))

        # Display the hand center
        if center:
            center_text = f"Center (x:{self.center[0]}, y:{self.center[1]})"
            cv2.circle(
                image,
                self.center,
                7,
                (0,255,0),
                -1
            )
            cv2.circle(
                image,
                self.center,
                7,
                (0,0,0),
                2
            )
            stack_text(
                image,
                [center_text],
                self.center,
                debug_font,
                debug_text_size * 1.5,
                debug_thickness * 2,
                (0,255,0),
                HAlign.LEFT,
                VAlign.BOTTOM,
                0
            )

        # Display the hand side ("Left" or "Right")
        if side:
            stack_text(
                image,
                [self.side],
                (self.wrist[0], self.box.opposite[1]),
                debug_font,
                debug_text_size * 2,
                debug_thickness * 2,
                (0,0,0),
                HAlign.CENTER,
                VAlign.TOP
            )

        # Display the hand flags and fingers up
        text_lines = []
        if flags:
            flag_text = f"Flags: {self.flags}"
            text_lines.append(flag_text)
        if fingers:
            text_lines.append("Fingers:")
            for finger in self.fingers_up():
                text_lines.append(finger)
        if self.side == "Left":
            stack_text(
                image,
                text_lines,
                (0,0),
                debug_font,
                debug_text_size,
                debug_thickness,
                (0,0,0)
            )
        else:
            stack_text(
                image,
                text_lines,
                (width, 0),
                debug_font,
                debug_text_size,
                debug_thickness,
                (0, 0, 0),
                h_align=HAlign.RIGHT
            )

        # Display the position values for each fingertip and the wrist
        if tip_points:
            for tip, color in zip(
                [
                    Hand.THUMB_TIP_ID,
                    Hand.INDEX_TIP_ID,
                    Hand.MIDDLE_TIP_ID,
                    Hand.RING_TIP_ID,
                    Hand.PINKY_TIP_ID,
                    Hand.WRIST_ID
                ],
                Hand.REGION_COLORS
            ):
                b, g, r = color
                b = b // 1.5
                g = g // 1.5
                r = r // 1.5
                x, y, _ = self.landmarks[tip]
                stack_text(
                    image,
                    [f"{self.landmarks[tip]}"],
                    (x, y),
                    debug_font,
                    debug_text_size,
                    debug_thickness,
                    (b, g, r),
                    HAlign.CENTER,
                    VAlign.BOTTOM
                )

    def set_connection_style(self, stroke=None, thickness=None):
        """
        Modifies the style of the hand connections when it is drawn.
        :param stroke: The BGR color of the connections
        :param thickness: The thickness of the connector lines in pixels
        """

        if stroke is not None:
            self.connection_style.stroke = stroke

        if thickness is not None:
            self.connection_style.thickness = int(thickness)

    def set_landmarks_style(self, fill=None, stroke=None, radius=None, thickness=None):
        """
        Modifies the style of the hand landmarks when drawn.
        :param fill: The BGR color of the landmarks
        :param stroke: The BGR color of the outline of the landmarks
        :param thickness: Thickness of outline on circle
        :param radius: The radius of the landmarks
        """
        if fill is not None:
            self.landmark_style.fill = fill
        if stroke is not None:
            self.landmark_style.stroke = stroke
        if radius is not None:
            self.landmark_style.radius = int(radius)
        if thickness is not None:
            self.landmark_style.thickness = int(thickness)


class HandDetector:
    def __init__(self, max_hands=2):
        with get_model_path("hand_landmarker.task") as model_path:
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=model_path
                ),
                num_hands=max_hands,
                running_mode=vision.RunningMode.VIDEO
            )
            self.landmarker = vision.HandLandmarker.create_from_options(
                options
            )
        self.timestamp_ms = 0
        self.frame_rate = 30

    def find_hands(self, image, flip=False):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp_ms
        )
        self.timestamp_ms += int(1000/self.frame_rate)

        if not result.hand_landmarks:
            return []

        hands = []

        for landmarks, handedness in zip(
            result.hand_landmarks,
            result.handedness
        ):
            pixel_landmarks = []
            x_list = []
            y_list = []
            for lm in landmarks:
                px_lm = (int(lm.x * width), int(lm.y * height), int(lm.z * width))
                pixel_landmarks.append(px_lm)
                x_list.append(int(lm.x * width))
                y_list.append(int(lm.y * height))
            bounding_box = BoundingBox(
                (min(x_list) - 10, min(y_list) - 10),
                (max(x_list) + 10, max(y_list) + 10)
            )
            category = handedness[0]   # usually one entry
            if flip:
                side = category.category_name   # "Left" or "Right"
            else:
                if category.category_name == "Left":
                    side = "Right"
                else:
                    side = "Left"
            pixel_landmarks = tuple(pixel_landmarks)
            hand = Hand(pixel_landmarks, landmarks, side, bounding_box, image)
            hands.append(hand)

        return hands

def main():
    # Initialize the webcam to capture video
    cap = cv2.VideoCapture(0)

    # Initialize the HandDetector class with the given parameters
    hand_detector = HandDetector(2)
    # Continuously get frames from the webcam
    while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()
        img = cv2.flip(img, 1)
        # Find hands in the current frame
        hands = hand_detector.find_hands(img)

        # Methods for each hand
        for hand in hands:
            hand.debug()

        # Display the image in a window
        cv2.imshow("Image", img)

        # Close the window if user presses 'q' or the X button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

if __name__ == "__main__":
    main()