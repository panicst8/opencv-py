""" opencv thingy """

import cv2
import numpy as np

img = cv2.imread("assets/finn.jpg")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(img, (7, 7), 0)

imgCanny = cv2.Canny(img, 100, 100)

kernal = np.ones((5, 5), np.uint8)

imgDilation = cv2.dilate(imgCanny, kernal, iterations=2)
imgEroded = cv2.erode(imgDilation, kernal, iterations=2)

cv2.imshow("Canny output", imgCanny)
cv2.waitKey(0)

cv2.imshow("dilation output", imgDilation)
cv2.waitKey(0)

cv2.imshow("eroded output", imgEroded)
cv2.waitKey(0)

quit()

# cv2.imshow("output", img)
# cv2.waitKey(0)

# cv2.imshow("blur output", imgBlur)
# cv2.waitKey(0)

# cv2.imshow("gray output", imgGray)
# cv2.waitKey(0)

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
