from dataclasses import dataclass
import cv2
import numpy as np
from enum import Enum, auto
import math

class HAlign(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()

class VAlign(Enum):
    TOP = auto()
    MIDDLE = auto()
    BOTTOM = auto()

@dataclass
class Landmark:
    """
    Stores landmark points, colors, and z-index when drawing them for hand connections.

    :ivar pos: The position of the landmark in 2D pixel space.
    :ivar z: The z-index of the landmark, used for sorting the order to draw them in.
    :ivar color: The color of the landmark.
    """
    pos: tuple[int, int]
    z: int
    color: tuple[int, int, int]

@dataclass
class Connection:
    """
    Stores line points, colors, and z-index when drawing them for hand connections.

    :ivar start: The start point of the line in 2D pixel space.
    :ivar end: The end point of the line in 2D pixel space.
    :ivar z: The average z-index of the line, used for sorting the order to draw them in.
    :ivar color: The color of the line.
    """
    start: tuple[int, int]
    end: tuple[int, int]
    z: float
    color: tuple[int, int, int]

@dataclass
class LandmarkStyle:
    """
    Stores landmark style fill color, stroke color, radius, and outline thickness.

    :ivar fill: BGR fill color of the landmarks.
    :ivar stroke: BGR stroke color of the landmarks.
    :ivar radius: Radius of landmarks in pixels.
    :ivar thickness: Thickness of outline of landmarks in pixels.
    """
    fill: tuple[int, int, int] = (0, 0, 255)
    stroke: tuple[int, int, int] = (255, 255, 255)
    radius: int = 5
    thickness: int = 1

@dataclass
class ConnectionStyle:
    """
    Stores connection style for stroke color and outline thickness.

    :ivar stroke: BGR stroke color of the connections.
    :ivar thickness: Thickness of connection in pixels.
    """
    stroke: tuple[int, int, int] = (255, 255, 255)
    thickness: int = 3

class Skeleton:
    """
    Stub for upcoming features.
    """
    def __init__(self, region_landmarks, region_connections, region_colors, landmark_style, connection_style):
        pass
    def draw(self, image):
        pass

class BoundingBox:
    """
    BoundingBox for displaying the height, width and position of objects identified on the screen.

    :ivar center: 2D pixel coordinates of the bounding box.
    :ivar box: Stores x, y, height, and width.
    :ivar width: Width of bounding box.
    :ivar height: Height of bounding box.
    :ivar size: Tuple containing width and height.
    :ivar origin: Top left corner of bounding box.
    :ivar opposite: Bottom right corner of bounding box.
    """
    def __init__(self, top_left_corner : tuple[int, int], bottom_right_corner : tuple[int, int]):
        """
        Initialize bounding box.

        :param top_left_corner:
        :param bottom_right_corner:
        """
        x, y = top_left_corner
        x2, y2 = bottom_right_corner
        w = abs(x2 - x)
        h = abs(y2 - y)
        # Integer pixel center of the bounding box
        self.center = (
            (x + x2) // 2,
            (y + y2) // 2
        )
        self.box : tuple[int, int, int, int] = (x, y, w, h)
        # Dimensions in pixels
        self.width : int = w
        self.height : int = h
        self.size : tuple[int, int]= (w, h)
        self.origin : tuple[int, int]= (x, y)
        self.opposite :tuple[int, int]= (x2, y2)

    def draw(self, image : np.ndarray, color: tuple[int, int, int]=(0,127,0), thickness: int=2):
        """
        Draw rectangle bounding box.

        :param image: Image to draw bounding box on.
        :param color: BGR color of bounding box outline.
        :param thickness: Thickness of line in pixels.
        """
        cv2.rectangle(
            image,
            self.origin,
            self.opposite,
            color,
            thickness
        )

    def draw_corners(self, image : np.ndarray, length: int=30, thickness:int=5, stroke:tuple[int, int, int]=(0, 255, 0)):
        """
        Draw corners of the bounding box.

        :param image: Image to draw bounding box on.
        :param length: Length of corner lines in pixels.
        :param thickness: Thickness of line in pixels.
        :param stroke: BGR color of corner lines.
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

def overlay_image(
    target: np.ndarray,
    image_path: str,
    origin: tuple[int, int],
    h_align: HAlign = HAlign.LEFT,
    v_align: VAlign = VAlign.TOP,
    scale: float = 1.0
):
    """
    Draw an image (png/jpg) onto a target image with alignment and transparency.

    :param target: BGR image to draw onto (modified in place)
    :param image_path: path to png/jpg (png supports alpha)
    :param origin: (x,y) coordinate of image placement
    :param h_align: horizontal alignment relative to (x, y)
    :param v_align: vertical alignment relative to (x, y)
    :param scale: scaling factor (1.0 = original size)
    """

    overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise ValueError(f"Could not load image: {image_path}")

    # --- scale image ---
    if scale != 1.0:
        h, w = overlay.shape[:2]
        new_w = max(1, int(w * scale))  # prevent 0-size
        new_h = max(1, int(h * scale))
        overlay = cv2.resize(
            overlay,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        )

    h, w = overlay.shape[:2]
    x, y = origin

    # --- alignment ---
    if h_align == HAlign.LEFT:
        ox = x
    elif h_align == HAlign.CENTER:
        ox = x - w // 2
    else:  # RIGHT
        ox = x - w

    if v_align == VAlign.TOP:
        oy = y
    elif v_align == VAlign.MIDDLE:
        oy = y - h // 2
    else:  # BOTTOM
        oy = y - h

    # --- clip to target bounds ---
    th, tw = target.shape[:2]

    x1 = max(0, ox)
    y1 = max(0, oy)
    x2 = min(tw, ox + w)
    y2 = min(th, oy + h)

    if x1 >= x2 or y1 >= y2:
        return

    overlay_x1 = x1 - ox
    overlay_y1 = y1 - oy
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    roi = target[y1:y2, x1:x2]
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    # --- alpha blending ---
    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha = alpha[..., None]

        roi[:] = (1 - alpha) * roi + alpha * overlay_crop[:, :, :3]
    else:
        roi[:] = overlay_crop[:, :, :3]

def rectangle(
    target: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    fill: tuple[int, int, int] | None = (255, 255, 255),
    outline: tuple[int, int, int] | None= (0, 0, 0),
    thickness: int = 1,

):
    """
    Draws a rectangle with simple parameters and default values.

    :param target: Target image to draw the shape on.
    :param top_left: (x,y) coordinate of top left corner.
    :param bottom_right: (x,y) coordinate of bottom right corner.
    :param fill: (B, G, R) color of fill. A value of None will remove the fill altogether
    :param outline: (B, G, R) color of outline. A value of None will remove the outline altogether.
    :param thickness: Thickness of outline. A value less than 1 will remove the outline altogether.
    :return: None
    """
    if fill is not None:
        cv2.rectangle(target, top_left, bottom_right, fill, -1)
    if outline is not None and thickness > 0:
        cv2.rectangle(target, top_left, bottom_right, outline, thickness)


def line(
    target: np.ndarray,
    *points: tuple[int, int] | list[tuple, tuple],
    color: tuple[int, int, int] = (0,0,0),
    thickness: int = 2,
    closed: bool = False
):
    """
    Draws a line with simple parameters and default values.

    :param target: Target image to draw the shape on.
    :param points: List of points to draw the line along. Can include more than 2 points, but must be at least 2 points.
    :param color:(B, G, R) color of outline. A value of None will remove the outline altogether.
    :param thickness: Thickness of outline. A value less than 1 will remove the outline altogether.
    :param closed: When True, connects the last point to the first point. When False, leaves shape open.
    :return: None
    """
    if len(points) == 1 and isinstance(points[0], (list, tuple)):
        points = points[0]
    if len(points) == 2 and isinstance(points[0], (list, tuple)):
        cv2.line(target, points[0], points[1], color, thickness)
    elif len(points) > 2:
        pts_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(target, [pts_array], closed, color, thickness)
    else:
        raise ValueError("You need at least 2 points to make a line. Points must be in the form (x, y)")

def ellipse(
        target: np.ndarray,
        center: tuple[int, int],
        radius: int,
        cross_radius: int | None = None,
        rotation: int = 0,
        start_angle: int = 0,
        end_angle: int = 360,
        fill: tuple[int, int, int] | None = (255, 255, 255),
        outline: tuple[int, int, int] | None = (0,0,0),
        thickness: int = 1,
        closed: bool=True
):
    """
    Draws an ellipse with simple parameters and default values.

    :param target: Target image to draw the shape on.
    :param center: (x,y) coordinate of center point.
    :param radius: Main radius of the ellipse.
    :param cross_radius: Cross radius of the ellipse. None results in a perfect circle.
    :param rotation: Rotation of ellipse in degrees.
    :param start_angle: Starting angle of arc in degrees.
    :param end_angle: Ending angle of arc in degrees.
    :param fill: (B, G, R) color of fill. A value of None will remove the fill altogether
    :param outline: (B, G, R) color of outline. A value of None will remove the outline altogether.
    :param thickness: Thickness of outline. A value less than 1 will remove the outline altogether.
    :param closed: If True, draws the outline along all edges of the arc. Otherwise, only draws the rounded section.
    :return: None
    """
    if cross_radius is None:
        cross_radius = radius
    if fill is not None:
        cv2.ellipse(target, center, (radius, cross_radius), rotation, start_angle, end_angle, fill, -1)
    if outline is not None and thickness > 0:
        cv2.ellipse(target, center, (radius, cross_radius), rotation, start_angle, end_angle, outline, thickness)
        closed = closed and (start_angle + end_angle) % 360 != 0
        if closed:
            cx, cy = center
            angles = [start_angle, end_angle]
            for angle in angles:
                theta = math.radians(angle)
                x = int(cx + radius * math.cos(theta))
                y = int(cy + radius * math.sin(theta))
                cv2.line(target, center, (x,y), outline, thickness)

def circle(
        target: np.ndarray,
        center: tuple[int, int],
        radius: int,
        fill: tuple[int,int,int] | None = (255, 255, 255),
        outline: tuple[int, int, int] | None = (0,0,0),
        thickness: int = 1
):
    """
    Draws a circle with simple parameters and default values.

    :param target: Target image to draw the shape on.
    :param center: (x,y) coordinate of center point.
    :param radius: Radius of circle.
    :param fill: (B, G, R) color of fill. A value of None will remove the fill altogether
    :param outline: (B, G, R) color of outline. A value of None will remove the outline altogether.
    :param thickness: Thickness of outline. A value less than 1 will remove the outline altogether.
    :return: None
    """
    if fill is not None:
        cv2.circle(target, center, radius, fill, -1)
    if outline is not None and thickness > 0:
        cv2.circle(target, center, radius, outline, thickness)

def polygon(
        target: np.ndarray,
        *points,
        fill: tuple[int, int, int] | None =(255, 255, 255),
        outline: tuple[int, int, int] | None =(0, 0, 0),
        thickness: int=2,
        closed: bool=True
):
    """
    Draws a polygon with simple parameters and default values.

    :param target: Target image to draw the shape on.
    :param points: List of points to draw the line along. Can include more than 3 points, but must be at least 3 points.
    :param fill: (B, G, R) color of fill. A value of None will remove the fill altogether
    :param outline: (B, G, R) color of outline. A value of None will remove the outline altogether.
    :param thickness: Thickness of outline. A value less than 1 will remove the outline altogether.
    :param closed: When True, connects the last point to the first point. When False, leaves shape open.
    :return: None
    """
    if len(points) == 1 and isinstance(points[0], (list, tuple)):
        points = points[0]

    if len(points) < 3:
        raise ValueError("Polygon requires at least 3 points")

    pts_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    if fill is not None:
        cv2.fillPoly(target, [pts_array], fill)

    if outline is not None and thickness > 0:
        cv2.polylines(target, [pts_array], closed, outline, thickness)


if __name__ == "__main__":
    WINDOW_NAME = "Camera Feed"
    WEBCAM = 0
    WIDTH = 1280
    HEIGHT = 720

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)
    cap = cv2.VideoCapture(WEBCAM)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    ###############################
    #  Your setup code goes here  #
    ###############################

    while True:

        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        ellipse(frame, (300, 300), 300, end_angle=180)
        line(frame, (1,1), (200, 200))
        ###############################################
        #  Your code that runs every frame goes here  #
        ###############################################

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()