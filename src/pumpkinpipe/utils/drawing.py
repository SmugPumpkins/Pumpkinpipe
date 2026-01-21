import cv2
from typing import Tuple

class BoundingBox:
    def __init__(self, top_left_corner : Tuple[int, int], bottom_right_corner : Tuple[int, int]):
        x, y = top_left_corner
        x2, y2 = bottom_right_corner
        w = abs(x2 - x)
        h = abs(y2 - y)
        # Integer pixel center of the bounding box
        self.center = (
            (x + x2) // 2,
            (y + y2) // 2
        )
        self.box = (x, y, w, h)
        # Dimensions in pixels
        self.width = w
        self.height = h
        self.size = (w, h)
        self.origin = (x, y)
        self.opposite = (x2, y2)

    def draw(self, image, color=(0,127,0), thickness=2):
        cv2.rectangle(
            image,
            self.origin,
            self.opposite,
            color,
            thickness
        )

    def draw_corners(self, image, length=30, thickness=5, stroke=(0, 255, 0)):
        """
        :param image: Image to draw on.
        :param length: length of the corner line
        :param thickness: thickness of the corner line
        :param stroke: Color of the Corners
        """
        x, y = self.origin
        x1, y1 = self.opposite
        # Top Left  x,y
        cv2.line(image, (x, y), (x + length, y), stroke, thickness)
        cv2.line(image, (x, y), (x, y + length), stroke, thickness)
        # Top Right  x1,y
        cv2.line(image, (x1, y), (x1 - length, y), stroke, thickness)
        cv2.line(image, (x1, y), (x1, y + length), stroke, thickness)
        # Bottom Left  x,y1
        cv2.line(image, (x, y1), (x + length, y1), stroke, thickness)
        cv2.line(image, (x, y1), (x, y1 - length), stroke, thickness)
        # Bottom Right  x1,y1
        cv2.line(image, (x1, y1), (x1 - length, y1), stroke, thickness)
        cv2.line(image, (x1, y1), (x1, y1 - length), stroke, thickness)


class LandmarkStyle:
    def __init__(self, fill=(0,0,255), stroke=(255,255,255), radius=5, thickness=1):
        self.fill = fill
        self.stroke = stroke
        self.radius = radius
        self.thickness = thickness

    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_stroke(self, stroke):
        self.stroke = stroke

    def set_radius(self, radius):
        self.radius = radius

    def set_fill(self, fill):
        self.fill = fill

class ConnectionStyle:
    def __init__(self, stroke=(255,255,255), thickness=3):
        self.stroke = stroke
        self.thickness = thickness

    def set_thickness(self, thickness):
        self.thickness = thickness

    def set_stroke(self, stroke):
        self.stroke = stroke


