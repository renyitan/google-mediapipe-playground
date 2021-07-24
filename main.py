import cv2
import mediapipe as mp
import time
import handDetector as detector

# activate webcam
WEBCAM_1, WEBCAM_2, WEBCAM_3 = 0, 1, 2
video_capture = cv2.VideoCapture(WEBCAM_1)

# initialise hands object
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# initialise drawing utility
mp_draw = mp.solutions.drawing_utils

def main():

    hand_detector = detector.handDetector()

    while True:
        success, image = video_capture.read()
        image = hand_detector.find_hands(image)
        landmarks = hand_detector.find_positions(image)

        if len(landmarks) != 0:
            print(landmarks[4])

        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()