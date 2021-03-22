""" opencv thingy """

import cv2
import numpy as np

img = cv2.imread("assets/finn.jpg")
cv2.imshow("output", img)
cv2.waitKey(0)

# image change to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray output", imgGray)
# cv2.waitKey(0)

# image blur
imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
# cv2.imshow("blur output", imgBlur)
# cv2.waitKey(0)

# resizing image
imgResize = cv2.resize(img, (300, 200))
print("img: ", img.shape, "\nimResize: ", imgResize.shape)
cv2.imshow("resized output", imgResize)
cv2.waitKey(0)

# crop image
imgCropped = img[0:500, 200:500]
cv2.imshow("cropped output", imgCropped)
cv2.waitKey(0)


# imgCanny = cv2.Canny(img, 100, 100)
# cv2.imshow("Canny output", imgCanny)
# cv2.waitKey(0)

# kernal = np.ones((5, 5), np.uint8)

# imgDilation = cv2.dilate(imgCanny, kernal, iterations=2)
# cv2.imshow("dilation output", imgDilation)
# cv2.waitKey(0)

# imgEroded = cv2.erode(imgDilation, kernal, iterations=2)
# cv2.imshow("eroded output", imgEroded)
# cv2.waitKey(0)

quit()


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
