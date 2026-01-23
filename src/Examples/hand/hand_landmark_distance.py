"""
An example of the landmark distance method for the hand detector module.
Author: Nathan Forsyth
"""
from pumpkinpipe.hand import HandDetector
import cv2

# Open a connection to the default webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detector with support for up to two hands
hand_detector = HandDetector(max_hands=2)

# Main capture loop
while True:
    # Read a single frame from the webcam
    success, frame = cap.read()

    # Mirror the frame horizontally for a natural, selfie-style view
    frame = cv2.flip(frame, 1)

    # Detect hands in the current frame
    hands = hand_detector.find_hands(frame)

    # For each detected hand, calculate the distance between two landmarks
    # Landmark indices 4 and 8 correspond to the thumb tip and index fingertip
    for hand in hands:
        # Computes the distance between the specified landmarks and,
        # when draw=True, visualizes the measurement directly on the frame
        hand.landmark_distance(4, 8, draw=True)

    # Display the processed frame with distance annotations
    cv2.imshow("Hand Distance Example", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit if the window is manually closed
    if cv2.getWindowProperty("Hand Distance Example", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
