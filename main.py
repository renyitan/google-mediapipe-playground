import cv2
import mediapipe as mp
import time

# use webcam 1
capture = cv2.VideoCapture(0)

while True:
    success, img = capture.read()



    cv2.imshow("Image", img)
    cv2.waitKey(1)
