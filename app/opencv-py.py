""" opencv thingy """

import cv2
import numpy as np


def skew() -> None:
    """ skew playing card """

    width, height = 250, 350

    pts_1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])
    pts_2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    img = cv2.imread("assets/cards.jpg")

    matrix = cv2.getPerspectiveTransform(pts_1, pts_2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imshow("Cards before skew", img)
    cv2.waitKey(0)

    cv2.imshow("Card skewed", imgOutput)
    cv2.waitKey(0)


skew()

quit()


def diy_images():
    img2 = np.zeros((512, 512, 3), np.uint8)
    # print(img2)

    # img2[:] = 255, 0, 0 # make image blue
    cv2.line(img2, (10, 10), (100, 100), (0, 255, 0), 5)
    cv2.rectangle(img2, (10, 10), (100, 100), (0, 255, 0), 5)
    cv2.circle(img2, (150, 150), 100, (0, 0, 255), cv2.FILLED)
    cv2.putText(
        img2, "Hello World", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2
    )

    cv2.imshow("Black Image", img2)
    cv2.waitKey(0)


img = cv2.imread("assets/finn.jpg")
# cv2.imshow("output", img)
# cv2.waitKey(0)

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
# print("img: ", img.shape, "\nimResize: ", imgResize.shape)
# cv2.imshow("resized output", imgResize)
# cv2.waitKey(0)

# crop image
imgCropped = img[0:500, 200:500]
# cv2.imshow("cropped output", imgCropped)
# cv2.waitKey(0)


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
