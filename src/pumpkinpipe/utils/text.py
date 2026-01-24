import cv2
from enum import Enum, auto

class HAlign(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()

class VAlign(Enum):
    TOP = auto()
    MIDDLE = auto()
    BOTTOM = auto()


def measure_text_block(lines, font, font_scale, thickness, margin=20):
    sizes = [cv2.getTextSize(t, font, font_scale, thickness) for t in lines]

    widths  = [w for (w, h), _ in sizes]
    heights = [h for (w, h), _ in sizes]
    baselines = [b for (_, _), b in sizes]

    block_width  = max(widths)
    block_height = sum(heights) + sum(baselines) + margin * (len(lines) - 1)

    return block_width, block_height, sizes


def block_x_offset(block_width, align: HAlign, margin=15):
    match align:
        case HAlign.LEFT:
            return margin
        case HAlign.CENTER:
            return -block_width // 2
        case HAlign.RIGHT:
            return -(block_width + margin)


def block_y_offset(block_height, align: VAlign, margin=15):
    match align:
        case VAlign.TOP:
            return margin
        case VAlign.MIDDLE:
            return -block_height // 2
        case VAlign.BOTTOM:
            return -(block_height + margin)



def stack_text(
    image,
    lines,
    origin,
    font,
    font_scale,
    thickness,
    color,
    h_align: HAlign = HAlign.LEFT,
    v_align: VAlign = VAlign.TOP,
    margin=5
):
    block_w, block_h, sizes = measure_text_block(
        lines, font, font_scale, thickness, margin
    )

    ox = origin[0] + block_x_offset(block_w, h_align)
    oy = origin[1] + block_y_offset(block_h, v_align)

    # baseline for first line (inside top margin)
    first_h = sizes[0][0][1]
    y_cursor = oy + first_h
    x = ox
    for text, ((w, h), baseline) in zip(lines, sizes):

        # consistent justification for all lines
        match h_align:
            case HAlign.LEFT:
                x = ox
            case HAlign.CENTER:
                x = ox + (block_w - w) // 2
            case HAlign.RIGHT:
                x = ox + (block_w - w)

        cv2.putText(
            image,
            text,
            (x, y_cursor),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

        # move DOWN for next line
        y_cursor += h + baseline + margin

def outline_text():
    # TODO: Create a method that allows for a user to input text with an outline color stacked on top of text without it
    pass

class TextBox:
    def __init__(self):
        pass

class CreditText:
    def __init__(self):
        pass
