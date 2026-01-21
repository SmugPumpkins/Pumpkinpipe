"""
Hand Detection Module
Author: Nathan Forsyth
"""
import cv2, math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pumpkinpipe.utils.model_loader import get_model_path
from pumpkinpipe.utils.drawing import BoundingBox, LandmarkStyle, ConnectionStyle
from pumpkinpipe.utils.text import stack_text, HAlign, VAlign
from mediapipe.tasks.python.vision import HandLandmarksConnections
connections = HandLandmarksConnections.HAND_CONNECTIONS

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



class Hand:
    def __init__(self, landmarks, original_landmarks, side, box : BoundingBox):
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
        self.thumb = self.landmarks[4]
        self.index = self.landmarks[8]
        self.middle = self.landmarks[12]
        self.ring = self.landmarks[16]
        self.pinky = self.landmarks[20]
        self.wrist = self.landmarks[0]
        self.connection_style = ConnectionStyle()
        self.landmark_style = LandmarkStyle()
        self.default_connection_style = ConnectionStyle()
        self.default_landmark_style = LandmarkStyle()
        self.flags = self.finger_flags()
        self.box = box
        self.center = self.box.center

    def landmark_distance(self, landmark_index_1, landmark_index_2, image=None):
        """
        Finds the distance in pixels between 2 specified landmarks.
        :param landmark_index_1: The landmark index for the first point
        :param landmark_index_2: The landmark index for the second point
        :param image: If not None, draws a line between the 2 points on the specified image
        :return: The distance between 2 points
        """
        landmark_1 = self.landmarks[landmark_index_1]
        landmark_2 = self.landmarks[landmark_index_2]
        distance : float = math.dist(landmark_1, landmark_2)
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
        distance_threshold = 0.31

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
        finger_indices = [8, 12, 16, 20]

        # Compare the distance between the tip and the wrist to the distance between the knuckle and the wrist
        for index in finger_indices:
            tip_distance = math.dist(self.landmarks[index], self.landmarks[0])
            knuckle_distance = math.dist(self.landmarks[index - 2], self.landmarks[0])
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

    def draw(self, image):
        """
        Draws the hand skeleton on the specified image.
        Currently, this library uses a custom way to draw while
        drawing_utils.py remains unimplemented in the
        official mediapipe release.
        :param image: The target image for the drawing.
        """
        for connection in connections:
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

    def debug(self, image, skeleton=True, bounding_box=True, center=True, side=True, fingers=True, flags=True):
        """
        Draws the requested debug information. Defaults to all debug information.
        """
        height, width, _ = image.shape
        debug_text_size = 1.2
        debug_font = cv2.FONT_HERSHEY_PLAIN
        debug_thickness = 1
        if skeleton:
            self.draw(image)
        if bounding_box:
            self.box.draw_corners(image, length=20, thickness=5, stroke=(0,0,0))
            self.box.draw_corners(image, length=19, thickness=4, stroke=(255,255,255))
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
                VAlign.BOTTOM
            )
        if side:
            stack_text(
                image,
                [self.side],
                (self.wrist[0], self.box.opposite[1]),
                debug_font,
                debug_text_size,
                debug_thickness,
                (0,0,0),
                HAlign.CENTER,
                VAlign.TOP
            )
        lines = []
        if flags:
            flag_text = f"Flags: {self.flags}"
            lines.append(flag_text)
        if fingers:
            lines.append("Fingers:")
            for finger in self.fingers_up():
                lines.append(finger)
        if self.side == "Left":
            stack_text(
                image,
                lines,
                (0,0),
                debug_font,
                debug_text_size,
                debug_thickness,
                (0,0,0)
            )
        else:
            stack_text(
                image,
                lines,
                (width, 0),
                debug_font,
                debug_text_size,
                debug_thickness,
                (0, 0, 0),
                h_align=HAlign.RIGHT
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
            self.connection_style.thickness = thickness

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
            self.landmark_style.radius = radius
        if thickness is not None:
            self.landmark_style.thickness = thickness


class HandDetector:
    def __init__(self, max_hands=2):
        with get_model_path("hand_landmarker.task") as model_path:
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=model_path
                ),
                num_hands=max_hands,
                running_mode=vision.RunningMode.VIDEO
            )

            self.landmarker = vision.HandLandmarker.create_from_options(
                options
            )
        self.timestamp_ms = 0
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
        self.timestamp_ms += 33

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
            hand = Hand(pixel_landmarks, landmarks, side, bounding_box)
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
            hand.debug(img)

        # Display the image in a window
        cv2.imshow("Image", img)

        # Close the window if user presses 'q' or the X button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

if __name__ == "__main__":
    main()