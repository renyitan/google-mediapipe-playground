import cv2
import numpy as np
import HandDetector as handDetector
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# image constants
WEBCAM_1, WEBCAM_2, WEBCAM_3 = 0, 1, 2
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
TRACKER_COLOR = (255, 0, 255)
TRACKER_RADIUS = 5

# MP hands landmark constants
THUMB_TIP_CONSTANT = 4
INDEX_FINGER_TIP_CONSTANT = 8

# activate webcam & set frame
video_capture = cv2.VideoCapture(WEBCAM_1)
video_capture.set(3, SCREEN_WIDTH)
video_capture.set(4, SCREEN_HEIGHT)

# initialise hand detector class
handDetector = handDetector.HandDetector(detection_confidence=0.7)

# initialise pycaw audio utilities
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
MIN_VOLUME, MAX_VOLUME, STEP = volume.GetVolumeRange()

# volume.SetMasterVolumeLevel(-20.0, None)

while True:
    success, image = video_capture.read()

    image = handDetector.find_hands(image)
    landmarks = handDetector.find_positions(image, draw=False)

    if len(landmarks) != 0:
        # get values for index finger and thumb
        index_finger_tip_id, index_finger_tip_x, index_finger_tip_y = landmarks[INDEX_FINGER_TIP_CONSTANT]
        thumb_tip_id, thumb_tip_x, thumb_tip_y = landmarks[THUMB_TIP_CONSTANT]
        center_x, center_y = (index_finger_tip_x + thumb_tip_x) // 2, (index_finger_tip_y + thumb_tip_y) // 2

        # draw circle tracker and line on index and thumb
        cv2.circle(image, (index_finger_tip_x, index_finger_tip_y), TRACKER_RADIUS, TRACKER_COLOR, cv2.FILLED)
        cv2.circle(image, (thumb_tip_x, thumb_tip_y), TRACKER_RADIUS, TRACKER_COLOR, cv2.FILLED)
        cv2.line(image, (index_finger_tip_x, index_finger_tip_y), (thumb_tip_x, thumb_tip_y), TRACKER_COLOR, 2)
        # draw circle tracker between index and thumb
        cv2.circle(image, (center_x, center_y), TRACKER_RADIUS, TRACKER_COLOR, cv2.FILLED)

        # determine length of line
        line_length = math.hypot(thumb_tip_x - index_finger_tip_x, thumb_tip_y - index_finger_tip_y)
        print(line_length)
        # map line length to volume range
        mapped_volume = np.interp(line_length, [5, 250], [MIN_VOLUME, MAX_VOLUME])

        volume.SetMasterVolumeLevel(mapped_volume, None)

    # output video
    cv2.imshow("Img", image)
    cv2.waitKey(1)  # 1 millisecond delay
