import cv2
import numpy as np
import HandDetector as handDetector

# image constants
WEBCAM_1, WEBCAM_2, WEBCAM_3 = 0, 1, 2
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
TRACKER_COLOR = (255, 0, 255)
TRACKER_RADIUS = 15

# MP hands landmark constants
THUMB_TIP_CONSTANT = 4
INDEX_FINGER_TIP_CONSTANT = 8

# activate webcam & set frame
video_capture = cv2.VideoCapture(WEBCAM_1)
video_capture.set(3, SCREEN_WIDTH)
video_capture.set(4, SCREEN_HEIGHT)

# initialise hand detector class
handDetector = handDetector.HandDetector(detection_confidence=0.7)

while True:
    success, image = video_capture.read()

    image = handDetector.find_hands(image)
    landmarks = handDetector.find_positions(image, draw=False)

    if len(landmarks) != 0:
        # get values for index finger and thumb
        index_finger_tip_id, index_finger_tip_x, index_finger_tip_y = landmarks[INDEX_FINGER_TIP_CONSTANT]
        thumb_tip_id, thumb_tip_x, thumb_tip_y = landmarks[THUMB_TIP_CONSTANT]

        # draw circle tracker on index and thumb
        cv2.circle(image, (index_finger_tip_x, index_finger_tip_y), TRACKER_RADIUS, TRACKER_COLOR, cv2.FILLED)
        cv2.circle(image, (thumb_tip_x, thumb_tip_y), TRACKER_RADIUS, TRACKER_COLOR, cv2.FILLED)

    # output video
    cv2.imshow("Img", image)
    cv2.waitKey(1)  # 1 millisecond delay
