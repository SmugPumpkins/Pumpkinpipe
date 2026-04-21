"""
An example of using an image overlay.
Author: Nathan Forsyth
"""
from pumpkinpipe.utils.drawing import overlay_image, HAlign, VAlign
import cv2

WINDOW_NAME = "Overlay Example"
WIDTH = 1280
HEIGHT = 720

# Open a connection to the default webcam (index 0)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# Main capture loop
while True:

    # Read a single frame from the webcam
    success, frame = cap.read()

    # Mirror the frame horizontally for a more natural, selfie-style view
    frame = cv2.flip(frame, 1)

    overlay_image(frame, "../images/stem_collegiate_logo.jpg", (0, HEIGHT), h_align=HAlign.LEFT, v_align=VAlign.BOTTOM, scale=0.15)

    # Display the processed frame in a window
    cv2.imshow(WINDOW_NAME, frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit the loop if the window is manually closed
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
