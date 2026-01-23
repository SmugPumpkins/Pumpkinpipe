"""
An example of rendering hands to a different image than the one the hands are detected in.
Author: Nathan Forsyth
"""

from pumpkinpipe.hand import HandDetector
import cv2
import numpy as np

# Open a connection to the default webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detector (up to two hands)
hand_detector = HandDetector(max_hands=2)

# Main capture loop
while True:
    # Read a frame from the webcam
    success, frame = cap.read()

    # Mirror the frame for a natural, selfie-style orientation
    frame = cv2.flip(frame, 1)

    # Detect hands in the current frame
    hands = hand_detector.find_hands(frame)

    # Get frame dimensions for creating a separate render target
    height, width, channels = frame.shape

    # Create a blank (black) background image
    background = np.zeros((height, width, channels), dtype=np.uint8)

    # Draw detected hands onto the separate background image
    for hand in hands:
        hand.draw(background)

    # Display the background image with only the drawn hands
    cv2.imshow("Hand On Different Image Example", background)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit if the window is manually closed
    if cv2.getWindowProperty("Hand On Different Image Example", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release webcam resources
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
