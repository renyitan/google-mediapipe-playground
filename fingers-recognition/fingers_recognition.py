import cv2
from detectors.HandDectector import HandDetector as HandDetector
import mediapipe as mp

# image constants
WEBCAM_1, WEBCAM_2, WEBCAM_3 = 0, 1, 2
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
TRACKER_COLOR = (255, 0, 255)
TRACKER_RADIUS = 5

# MP hands landmark positions (https://google.github.io/mediapipe/solutions/hands.html#hand-landmark-model)
positions = mp.solutions.hands.HandLandmark

# activate webcam & set frame
video_capture = cv2.VideoCapture(WEBCAM_1)
video_capture.set(3, SCREEN_WIDTH)
video_capture.set(4, SCREEN_HEIGHT)

# initialise hand detector class
handDetector = HandDetector.HandDetector(detection_confidence=0.7)

while True:
    success, image = video_capture.read()
    image = handDetector.find_hands(image)
    landmarks = handDetector.find_positions(image, draw=False)

    finger_tip_positions = [positions.THUMB_TIP, positions.INDEX_FINGER_TIP, positions.MIDDLE_FINGER_TIP,
                            positions.RING_FINGER_TIP, positions.PINKY_TIP]
    finger_dip_positions = [positions.THUMB_IP, positions.INDEX_FINGER_DIP, positions.MIDDLE_FINGER_DIP,
                            positions.RING_FINGER_DIP, positions.PINKY_DIP]

    if len(landmarks) != 0:

        fingers = []

        flipped_handedness = handDetector.find_handedness(image)

        # 1 hand only
        # set default to "left hand"
        which_hand = "right" if flipped_handedness[0]['label'].lower() == "left" else "left"
        # print(which_hand)
        thumb_tip = landmarks[positions.THUMB_TIP]  # [id, x, y]
        if which_hand == "left":
            if thumb_tip[1] < landmarks[positions.INDEX_FINGER_TIP][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        if which_hand == "right":
            if thumb_tip[1] > landmarks[positions.INDEX_FINGER_TIP][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # for four fingers excluding thumb
        for id in range(1, len(finger_tip_positions)):
            # get current finger landmark data
            finger_tip = landmarks[finger_tip_positions[id]]  # [id, x, y]
            finger_dip = landmarks[finger_dip_positions[id]]

            if finger_tip[2] < finger_dip[2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)
        print(total_fingers)
        cv2.putText(image, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (51, 52, 210), 15)

    # output video
    cv2.imshow("Img", image)
    cv2.waitKey(1)  # 1 millisecond delay
