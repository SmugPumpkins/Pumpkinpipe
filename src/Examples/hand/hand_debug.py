"""
An example of the debug method for the hand module that shows off the different attributes that can be used with hands that are being tracked.
Author: Nathan Forsyth
"""
from pumpkinpipe.hand import HandDetector
import cv2

# Open a connection to the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Create a HandDetector instance
# max_hands limits how many hands the detector will attempt to track per frame
hand_detector = HandDetector(max_hands=2)

# Main capture loop
while True:

    # Read a single frame from the webcam
    success, frame = cap.read()

    # Mirror the frame horizontally for a more natural, selfie-style view
    frame = cv2.flip(frame, 1)

    # Run hand detection on the current frame
    # Returns a list of Hand objects detected in the image
    hands = hand_detector.find_hands(frame)

    # Iterate through all detected hands
    for hand in hands:
        # Draws debug visuals (landmarks, connections, bounding boxes, etc.)
        # directly onto the frame for visualization purposes
        hand.debug()

    # Display the processed frame in a window
    cv2.imshow("Hand Debug Example", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit the loop if the window is manually closed
    if cv2.getWindowProperty("Hand Debug Example", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
