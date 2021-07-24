import cv2
import mediapipe as mp
import time

# webcam constants
WEBCAM_1, WEBCAM_2, WEBCAM_3 = 0, 1, 2

class handDetector():
    def __init__(self, mode=False, max_num_hands=2,
                 detection_confidence=0.5, tracking_confidence=0.5
                 ):
        self.mode = mode
        self.max_hands = max_num_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # initialise hands object
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         self.detection_confidence,
                                         self.tracking_confidence)

        # initialise drawing utility
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        # convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:

                if draw:
                    self.mp_draw.draw_landmarks(image, hand_landmark,
                                                self.mp_hands.HAND_CONNECTIONS)

        return image

    def find_positions(self, image, hand_number=0, draw=True):
        landmarks = []

        # check if landmarks exists
        if self.results.multi_hand_landmarks:
            # select specific hand
            selected_hand = self.results.multi_hand_landmarks[hand_number]

            for id, landmark in enumerate(selected_hand.landmark):

                height, width, center = image.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)

                # adds id and position coordinates to list
                landmarks.append([id, center_x, center_y])

                if draw:
                    cv2.circle(image, (center_x, center_y), 5, (255, 0, 255), cv2.FILLED)

        return landmarks


def main():
    # activate webcam
    video_capture = cv2.VideoCapture(WEBCAM_1)

    hand_detector = handDetector()

    while True:
        success, image = video_capture.read()
        image = hand_detector.find_hands(image)
        landmarks = hand_detector.find_positions(image)

        if len(landmarks) != 0:
            print(landmarks)

        cv2.imshow("Image", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
