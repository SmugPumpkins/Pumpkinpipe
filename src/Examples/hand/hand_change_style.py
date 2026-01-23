"""
An example of changing the hand styles when rendering hands with the hand detector module.
Author: Nathan Forsyth
"""
from pumpkinpipe.hand import HandDetector
import cv2

# Open a connection to the default webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detector with support for up to two hands
hand_detector = HandDetector(max_hands=2)

# Thickness animation settings
# thick controls line thickness for hand connections
# thick_increment determines how fast thickness changes per frame
thick_increment = 0.5
thick = 1
MAX_THICK = 20

# Grayscale value animation settings
# value is used for stroke/fill intensity
# value_increment determines how fast the value changes per frame
value_increment = 5
value = 0
MAX_VALUE = 255

# Main capture loop
while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    # Mirror the frame for a natural viewing experience
    frame = cv2.flip(frame, 1)

    # Detect hands in the current frame
    hands = hand_detector.find_hands(frame)

    # Update grayscale value and reverse direction at bounds
    value += value_increment
    if value <= 0 or value >= MAX_VALUE:
        value_increment *= -1

    # Update thickness value and reverse direction at bounds
    thick += thick_increment
    if thick <= 1 or thick >= MAX_THICK:
        thick_increment *= -1

    # Apply different visual styles to each detected hand
    for i, hand in enumerate(hands):
        # Left hand: animated line thickness and landmark size
        # This is done by changing styles with hand.set_connection_style() and
        # hand.set_landmark_style()
        if hand.side == "Left":
            hand.set_connection_style(thickness=thick)
            hand.set_landmarks_style(radius=(3 + thick))

        # Right hand: animated grayscale stroke and inverted fill
        # This is done by changing styles with hand.set_connection_style() and
        # hand.set_landmark_style()
        if hand.side == "Right":
            hand.set_connection_style(stroke=(value, value, value), thickness=8)
            hand.set_landmarks_style(
                fill=(value, value, value),
                stroke=(255 - value, 255 - value, 255 - value),
                radius=8
            )

        # Draw the styled hand landmarks and connections onto the frame
        hand.draw()

    # Display the processed frame
    cv2.imshow("Hand Style Example", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit if the window is closed
    if cv2.getWindowProperty("Hand Style Example", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release webcam resources
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

