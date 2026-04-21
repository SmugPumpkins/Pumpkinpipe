import cv2
from enum import Enum, auto

import numpy as np


class HAlign(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()

class VAlign(Enum):
    TOP = auto()
    MIDDLE = auto()
    BOTTOM = auto()


def get_text_block_size(lines: list[str], font: int, font_scale: float, thickness: int, margin=20) -> tuple[int, int, list]:
    """
    Get the width and height of a list of lines of text.

    :param lines: List of text values as strings.
    :param font: Font to measure
    :param font_scale: Font scale to measure
    :param thickness: Thickness to measure
    :param margin: Margin to consider in measurement
    :return: Width, height, and list of sizes for each line of text
    """
    sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t in lines]

    widths  = [w for (w, h), _ in sizes]
    heights = [h for (w, h), _ in sizes]
    baselines = [b for (_, _), b in sizes]

    block_width  = max(widths)
    block_height = sum(heights) + sum(baselines) + margin * (len(lines) - 1)

    return block_width, block_height, sizes

def get_single_line_size(text: str, font: int, font_scale: float, thickness: int) -> tuple[int, int]:
    """
    Get the width and height of a single line of text.

    :param text: Text to measure
    :param font: Font to measure
    :param font_scale: Font scale to measure
    :param thickness: Font thickness to measure
    :return: Width and height as integers
    """
    size = cv2.getTextSize(text, font, font_scale, thickness)
    (width, height), baseline = size
    total_height = height + baseline
    return width, total_height

def align_single_line(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale: float = 1,
    thickness: int = 1,
    color=(255, 255, 255),
    h_align: HAlign = HAlign.LEFT,
    v_align: VAlign = VAlign.TOP,
    calculated_thickness=1
):
    """
    Display a single line of aligned text.

    :param image: Target image to display text
    :param text: String value of text
    :param origin: (x,y) coordinates of text
    :param font: Display font
    :param font_scale: Display scale
    :param thickness: Display thickness
    :param color: Display color
    :param h_align: Horizontal align setting
    :param v_align: Vertical align setting
    :param calculated_thickness: Thickness to use in size calculations (required for outlined text)
    :return: None
    """
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, calculated_thickness)

    ox, oy = origin

    if h_align == HAlign.LEFT:
        x = ox
    elif h_align == HAlign.CENTER:
        x = ox - w // 2
    else:
        x = ox - w

    if v_align == VAlign.TOP:
        y = oy + h
    elif v_align == VAlign.MIDDLE:
        y = oy + h // 2
    else:
        y = oy

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def get_text_x_offset(block_width, align: HAlign, margin=15) -> int:
    """
    Get the x offset for text based off of the horizontal alignment.

    :param block_width: Width of text block
    :param align: Alignment setting
    :param margin: Distance between text
    :return: X offset for text
    """
    match align:
        case HAlign.LEFT:
            return margin
        case HAlign.CENTER:
            return -block_width // 2
        case HAlign.RIGHT:
            return -(block_width + margin)


def get_text_y_offset(block_height, align: VAlign, margin=15) -> int:
    """
    Get the y offset for text based off of the vertical alignment.

    :param block_height: Height of text block
    :param align: Alignment setting
    :param margin: Distance between text
    :return: Y offset for text
    """
    match align:
        case VAlign.TOP:
            return margin
        case VAlign.MIDDLE:
            return -block_height // 2
        case VAlign.BOTTOM:
            return -(block_height + margin)



def stack_text(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    font = cv2.FONT_HERSHEY_PLAIN,
    font_scale: float = 1,
    thickness: int = 1,
    color = (255, 255, 255),
    h_align: HAlign = HAlign.LEFT,
    v_align: VAlign = VAlign.TOP,
    margin=5
):
    """
    Add multiple strings as separate lines to the image

    :param image: Target image to place text.
    :param lines: List of strings, each element is put on its own line.
    :param origin: (x,y) Coordinates for the origin of the font.
    :param font: Display font of text.
    :param font_scale: Display scale of text.
    :param thickness: Display line thickness of text.
    :param color: Display color of text. Defaults to white.
    :param h_align: Horizontal alignment of text.
    :param v_align: Vertical alignment of text.
    :param margin: Distance between lines of text in pixels.
    :return: None
    """
    block_w, block_h, sizes = get_text_block_size(
        lines, font, font_scale, thickness, margin
    )

    ox = origin[0] + get_text_x_offset(block_w, h_align)
    oy = origin[1] + get_text_y_offset(block_h, v_align)
    display_colors = ((0,0,0), color)
    outline_thickness = 5
    # baseline for first line (inside top margin)
    first_h = sizes[0][0][1]
    y_cursor = oy + first_h
    x = ox

    for text, ((w, h), baseline) in zip(lines, sizes):
        match h_align:
            case HAlign.LEFT:
                x = ox
            case HAlign.CENTER:
                x = ox + (block_w - w) // 2
            case HAlign.RIGHT:
                x = ox + (block_w - w)
        for i, col in enumerate(display_colors):
            cv2.putText(
                image,
                text,
                (x, y_cursor),
                font,
                font_scale,
                col,
                thickness + ((1-i) * outline_thickness),
                cv2.LINE_AA
            )
        y_cursor += h + baseline + margin

def outline_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        font=cv2.FONT_HERSHEY_PLAIN,
        font_scale:float =1,
        thickness:int =1,
        color=(255, 255, 255),
        h_align: HAlign = HAlign.LEFT,
        v_align: VAlign = VAlign.TOP,
        outline_color=(0,0,0),
        outline_thickness=5
):
    """
    Creates an outlined line of text.

    :param image: Target image to draw the text.
    :param text: String value of text to display
    :param origin: (x,y) coordinates of text
    :param font: Font to display
    :param font_scale: Scale of text
    :param thickness: Thickness of inner text
    :param color: Fill color of text
    :param h_align: Horizontal alignment
    :param v_align: Vertical alignment
    :param outline_color: Outline color
    :param outline_thickness: Outline thickness
    :return: None
    """
    align_single_line(image, text, origin, font, font_scale, outline_thickness, outline_color, h_align, v_align, thickness)
    align_single_line(image, text, origin, font, font_scale, thickness, color, h_align, v_align, thickness)

class TextBox:
    def __init__(self):
        pass

class CreditText:
    def __init__(self):
        pass
