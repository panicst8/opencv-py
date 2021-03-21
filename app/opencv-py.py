""" opencv thingy """

import cv2

img = cv2.imread("assets/finn.jpg")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("output", img)
cv2.waitKey(5000)

cv2.imshow("gray output", imgGray)

cv2.waitKey(5000)

cap = cv2.VideoCapture("assets/finn.mp4")

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)  # brightness setting

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
